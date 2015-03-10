

__kernel void colorConv(__global uchar * src, __global  float * dst, uint inByteStride, uint outStride)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    const int inPos = X * 3 /* components */ + Y * inByteStride;
    const int outPos = X + Y * outStride;

    float4 df = convert_float4(vload4(0, src + inPos));
    df.s3 = (float)(0);
    dst[outPos] = dot((float4)(0.299f, 0.587f, 0.114f, 0.0f), df);
}

__kernel void scaleDown(__global float * src, __global  float * dst, uint inStride, uint outStride)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    const int inPos = X * 2 + Y * 2 * inStride;
    const int outPos = X + Y * outStride;

    float4 sample;
    sample.s01 = vload2(0, src + inPos);
    sample.s23 = vload2(0, src + inPos + inStride);
    dst[outPos] = dot(sample, (float4)(0.25));
}

// call with work size dst.w dst.h
__kernel void scaleDownH(__global float * src, __global  float * dst, uint inStride, uint outStride)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    const int inPos = X * 2 + Y * inStride;
    const int outPos = X + Y * outStride;

    float2 sample;
    sample.s01 = vload2(0, src + inPos);
    dst[outPos] = dot(sample, (float2)(0.5));
}

// call once per input pixel (being w,h)
__kernel void scaleUpLinHPrysm(__global float * data, uint stride, uint offs, int w)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    
    const int inPos = offs + X + Y * stride;
    const int outPos = offs + w + X * 2 + Y * stride;
    
    float2 d = vload2(0, data + inPos);
    data[outPos] = d.s0;
    data[outPos + 1] = dot(d, (float2)(0.5f, 0.5f));
}

__kernel void scaleDownHPrysm(__global float * data, uint stride, uint offs, int w)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    const int inPos = offs + X * 2 + Y * stride;
    const int outPos = offs + w + X + Y * stride;

    float2 sample;
    sample.s01 = vload2(0, data + inPos);
    data[outPos] = dot(sample, (float2)(0.5));
}

__kernel void scaleUpH(__global float * src, __global  float * dst, uint inStride, uint outStride)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    float2 d = vload2(0,src + X + Y * inStride);
    const uint outPos = X * 2 + Y * outStride;
    dst[outPos] = d.s0;
    dst[outPos + 1] = dot(d, (float2)(0.5, 0.5));
}

// call once per output pixel (being w*2,h)
__kernel void scaleUpH_NN(__global float * src, __global  float * dst, uint inStride, uint outStride)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    float d = src [X + Y * inStride];
    const uint outPos = X * 2 + Y * outStride;
    vstore2((float2)(d,d), 0, dst + outPos);    
}

// call once per input pixel (being w,h*2)
__kernel void scaleUpV(__global float * src, __global  float * dst, uint inStride, uint outStride)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    const uint inPos = X + Y * inStride;
    float2 d;
    d.s0 = src[inPos];
    d.s1 = src[inPos + inStride];
    const uint outPos = X + 2 * Y * outStride;
    dst[outPos] = d.s0;
    dst[outPos + outStride] = dot(d, (float2)(0.5, 0.5));
}

// In short: X represents X coordinate on left image, Z represents X coordinate on right image, Y is Y coordinate for both
// Each execution is a match between left image at X,Y and right image at Z,Y
// We're addressing a specific branch of stereo vision, in which the two objectives, sightlines are parallel 
//
//    \     |\    /|     /
//     \    | \  / |    /
//      \   |  \/  |   /
//       \  |  /\  |  /
//        \ | /  \ | /
//         \|/    \|/
//          o      o
//        L          R
// 
// in this instance, FOVs lines are parallel, so it is ALWAYS the case that a point in L image is more to the right than in R;
// we can exploit the otherwise wasted tests to accomodate more useful work together, and shape a better work items size.  
//
// 
//     <-----  workload size 0 (X) = img W + 1   ----->
//  A B
//||7 8 ||0 1 | 0 2 | 0 3 | 0 4 | 0 5 | 0 6 | 0 7 | 0 8 |    ^
//| 7 8 | 0 1 ||1 2 | 1 3 | 1 4 | 1 5 | 1 6 | 1 7 | 1 8 |    |
//| 7 8 | 0 1 | 0 2 ||2 3 | 2 4 | 2 5 | 2 6 | 2 7 | 2 8 |    |  workload size 2 (Z) = img W / 2
//| 7 8 | 0 1 | 0 2 | 0 3 ||3 4 | 3 5 | 3 6 | 3 7 | 3 8 |    v
//            ^ 
//            |                 ||4 5 | 4 6 | 4 7 | 4 8 |
//             ----------             ||5 6 | 5 7 | 5 8 |
//                                          ||6 7 | 6 8 |
//                                                ||7 8 |
//
// 
// Each test results in an error value (among which we want to chose the lowest on Z, (fixed X, Y ))
// We can:
// 1) pack error and displacement/original position in an integer such that error occupies MSBs and atomic_min this packed value with the one at X,Y in result buffer
//		(error computation can include distance traveled, so that nearest pixel is chosen when several share the same error as originally computed)
// 2) store whole XxYxZ error array for portions (i.e. horizontal slices) of the input image, and operate more complex math to choose best match
// 
__kernel void blockMatch3x3W(__global float * srcL, __global float * srcR, __global  uint * inPrevDisparity, __global  uint * outDisparity, uint inStrideL, uint inStrideR, uint inPrevDisparityStride, uint outDisparityStride, uint N)
{
    const uint X = get_global_id(0);
    const uint Y = get_global_id(1);
    const uint Z = get_global_id(2);
    
    uint Rx, Lx;
    
    if (X<=Z)
    {
      Rx = N - Z;
      Lx = N + 1 - X; 
    }
    else
    {
      Rx = Z;
      Lx = X;
    }
    
    const uint inPosR = Rx + Y * inStrideR;
    const uint inPosL = Lx + Y * inStrideL;
    
    const float4 W[3] = { 
        (float4)(0.07511360795411207, 0.12384140315297386, 0.07511360795411207, 0) * 255, 
        (float4)(0.12384140315297386, 0.20417995557165622, 0.12384140315297386, 0) * 255, 
        (float4)(0.07511360795411207, 0.12384140315297386, 0.07511360795411207, 0) * 255
        }; // normalized weight matrix (gaussian(rho^2=1) for now); TODO: normalize to 256
    
    float4 br[3];
    float4 bl[3];
    
    // load Right Block from R src
    br[0] = vload4(0, srcR + inPosR);
    br[1] = vload4(0, srcR + inPosR + inStrideR);
    br[2] = vload4(0, srcR + inPosR + inStrideR * 2);
    
    // load Left Block from L src    
    bl[0] = vload4(0, srcL + inPosL);
    bl[1] = vload4(0, srcL + inPosL + inStrideL);
    bl[2] = vload4(0, srcL + inPosL + inStrideL * 2);
    
    // read old estimated disparity for X,Y and related error
    uint oldErrDisp = inPrevDisparity[X / 2 + Y * inPrevDisparityStride];
    // now: oldErrDisp.lo = disparity;
    //      oldErrDisp.hi = err;
    // if Z is (X+disp)*2 or (X+disp)*2 + 1 then err is oldErrDisp.hi, else err*2.
    
    uint disp = oldErrDisp & 0xFFFF;
    uint err = ((Z==(X+disp)*2) || (Z==(X+disp)*2 + 1))? oldErrDisp>>16 : (oldErrDisp>>16)<<1;
    
    uint outPos = X + Y * outDisparityStride;
    
    atomic_min(outDisparity + outPos,  // weighted sum of absolute differences
                        err + convert_uint_sat(dot(fabs(br[0]-bl[0]),W[0])+
                                        dot(fabs(br[1]-bl[1]),W[1])+
                                        dot(fabs(br[2]-bl[2]),W[2])) << 16 + (Lx - Rx)); // TODO: subtract averages, maybe
}


//// call once per 2 input pixel (being w/8,h)
//// the image has to be subdivisible by 2 three times, so it must be multiple of 8 (or it is cut)
//// image is subsampled 3 times horizontally, and supersampled once or twice so we have 5-6 levels
//__kernel void pyr(__global float * src, __global  float * dst, uint inStride, uint outStride)
//{
//    const uint X = get_global_id(0);
//    const uint Y = get_global_id(1);
//    
//    float4 d = vload4(0, src + X * 4 + Y * inStride);
//    
//    const uint outPos = X * 2 + Y * outStride;
//    dst[outPos] = d.s0;
//    dst[outPos + 1] = dot(d, (float2)(0.5, 0.5));
//}
