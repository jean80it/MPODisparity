using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MPODisparity
{
    class CLStuff
    {
        static readonly IntPtr intPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
        static readonly IntPtr intSize = (IntPtr)Marshal.SizeOf(typeof(int));
        static readonly IntPtr floatSize = (IntPtr)Marshal.SizeOf(typeof(float));

        Device _device;
        Context _context;
        OpenCL.Net.Program _program;
        ErrorCode err;
        CommandQueue _commandsQueue;
        Dictionary<string, Kernel> _kernels = new Dictionary<string, Kernel>();

        public CLStuff()
        {
            init(@"..\..\stuff.cl");
        }

        public void Dispose()
        {
            destroy();
        }

        #region openCL specific stuff

        private void assert(ErrorCode err, string message)
        {
            if (err != ErrorCode.Success)
            {
                throw new Exception("error:" + err + "; " + message);
            }
        }

        private void ContextNotify(string errInfo, byte[] data, IntPtr cb, IntPtr userData)
        {
            Debug.WriteLine("OpenCL Notification: " + errInfo);
        }

        private void init(string oclProgramSourcePath)
        {
            string kernelSource = File.ReadAllText(oclProgramSourcePath);

            string[] kernelNames = new string[] { "colorConv", "scaleDown", "scaleDownH", "scaleUpH", "scaleUpV", "scaleDownHPrism", "scaleUpLinHPrism", "blockMatch3x3W" };

            bool gpu = true;
            //err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL); 
            // NVidia driver doesn't seem to support a NULL first param (properties)
            // http://stackoverflow.com/questions/19140989/how-to-remove-cl-invalid-platform-error-in-opencl-code

            // now get all the platform IDs
            Platform[] platforms = Cl.GetPlatformIDs(out err);
            assert(err, "Error: Failed to get platform ids!");

            InfoBuffer deviceInfo = Cl.GetPlatformInfo(platforms[0], PlatformInfo.Name, out err);
            assert(err, "error retrieving platform name");
            Console.WriteLine("Platform name: {0}\n", deviceInfo.ToString());


            //                                 Arbitrary, should be configurable
            Device[] devices = Cl.GetDeviceIDs(platforms[0], gpu ? DeviceType.Gpu : DeviceType.Cpu, out err);
            assert(err, "Error: Failed to create a device group!");

            _device = devices[0]; // Arbitrary, should be configurable

            deviceInfo = Cl.GetDeviceInfo(_device, DeviceInfo.Name, out err);
            assert(err, "error retrieving device name");
            Debug.WriteLine("Device name: {0}", deviceInfo.ToString());

            deviceInfo = Cl.GetDeviceInfo(_device, DeviceInfo.ImageSupport, out err);
            assert(err, "error retrieving device image capability");
            Debug.WriteLine("Device supports img: {0}", (deviceInfo.CastTo<Bool>() == Bool.True));

            // Create a compute context 
            //
            _context = Cl.CreateContext(null, 1, new[] { _device }, ContextNotify, IntPtr.Zero, out err);
            assert(err, "Error: Failed to create a compute context!");

            // Create the compute program from the source buffer
            //
            _program = Cl.CreateProgramWithSource(_context, 1, new[] { kernelSource }, new[] { (IntPtr)kernelSource.Length }, out err);
            assert(err, "Error: Failed to create compute program!");

            // Build the program executable
            //
            err = Cl.BuildProgram(_program, 1, new[] { _device }, string.Empty, null, IntPtr.Zero);
            assert(err, "Error: Failed to build program executable!");
            InfoBuffer buffer = Cl.GetProgramBuildInfo(_program, _device, ProgramBuildInfo.Log, out err);
            Debug.WriteLine("build success: {0}", buffer.CastTo<BuildStatus>() == BuildStatus.Success);

            foreach (string kernelName in kernelNames)
            {
                // Create the compute kernel in the program we wish to run
                //
                OpenCL.Net.Kernel kernel = Cl.CreateKernel(_program, kernelName, out err);
                assert(err, "Error: Failed to create compute kernel!");
                _kernels.Add(kernelName, kernel);
            }

            // Create a command queue
            //
            _commandsQueue = Cl.CreateCommandQueue(_context, _device, CommandQueueProperties.None, out err);
            assert(err, "Error: Failed to create a command commands!");
        }

        public void finish()
        {
            Cl.Finish(_commandsQueue);
        }

        public void flush()
        {
            Cl.Flush(_commandsQueue);
        }

        private void destroy()
        {
            finish();

            //Clean up memory
            foreach (KeyValuePair<string, Kernel> k in _kernels)
            {
                Cl.ReleaseKernel(k.Value);
            }

            Cl.ReleaseCommandQueue(_commandsQueue);
            Cl.ReleaseProgram(_program);
            Cl.ReleaseContext(_context);

        }

        #endregion // openCL specific stuff

        private void invokeKernel(Kernel kernel, int[] offs, int[] range, params object[] additionalParams)
        {
            
            uint idx = 0;

            List<Tuple<IMap, IMem>> readBack = new List<Tuple<IMap, IMem>>();

            foreach (object o in additionalParams)
            {
                if (o.GetType().IsPrimitive)
                {
                    // set parameter
                    err = Cl.SetKernelArg(kernel, idx, new IntPtr(System.Runtime.InteropServices.Marshal.SizeOf(o)), o);
                    assert(err, string.Format("error on idx {0}", idx));
                    ++idx;
                }
                else
                {
                    IBufParam p = o as IBufParam;
                    if (p != null)
                    {
                        IMap map = p.Buf as IMap;
                        if (map != null)
                        {

                            MemFlags memFlags = MemFlags.None;

                            switch (p.Scope & (BufParamScope.InOut))
                            {
                                case BufParamScope.In:
                                    memFlags = MemFlags.ReadOnly;
                                    break;
                                case BufParamScope.Out:
                                    memFlags = MemFlags.WriteOnly;
                                    break;
                                case BufParamScope.InOut:
                                    memFlags = MemFlags.ReadWrite;
                                    break;
                            }

                            // buffer created here regardless of purpose (in/out)
                            IMem cl_buf = Cl.CreateBuffer(_context, memFlags, map.ByteSize, out err);

                            if ((p.Scope & BufParamScope.In) != 0)
                            {
                                // enqueue write or read, set flags as readonly, writeonly or both, depending on Flags
                                map.EnqueueWrite(_commandsQueue, cl_buf);
                            }

                            if ((p.Scope & BufParamScope.Out) != 0)
                            {
                                readBack.Add(Tuple.Create(map, cl_buf));
                            }

                            // set pointer as parameter here regardless of purpose (in/out)
                            err = Cl.SetKernelArg(kernel, idx, intPtrSize, cl_buf);
                            assert(err, string.Format("error on idx {0}", idx));
                            ++idx;
                        }
                    }
                    else
                    { 
                        IMem mem = o as IMem;
                        if (mem != null)
                        {
                            err = Cl.SetKernelArg(kernel, idx, intPtrSize, mem);
                            assert(err, string.Format("error on idx {0}", idx));
                            ++idx;
                        }
                    }
                }
            }

            IntPtr[] offsA = new IntPtr[offs.Length];
            for (int i = 0; i < offs.Length; ++i) { offsA[i] = new IntPtr(offs[i]); }
            IntPtr[] rangeA = new IntPtr[range.Length];
            for (int i = 0; i < range.Length; ++i) { rangeA[i] = new IntPtr(range[i]); }

            Event clevent;
            // execute
            err = Cl.EnqueueNDRangeKernel(_commandsQueue, kernel, 2, offsA, rangeA, null, 0, null, out clevent);
            clevent.Dispose();
            assert(err, "Cl.EnqueueNDRangeKernel");

            // read from output memory object into buffers for IMaps marked as output
            foreach (var r in readBack)
            {
                r.Item1.EnqueueRead(_commandsQueue, r.Item2);
            }

            // sync
            err = Cl.Finish(_commandsQueue);
            assert(err, "Cl.Finish");

            foreach (var r in readBack)
            {
                // release memory for buffers
                err = Cl.ReleaseMemObject(r.Item2);
                assert(err, "releasing mapBuffer");
            }
        }

        private void invokeKernelNoSync(Kernel kernel, int[] offs, int[] range, params object[] additionalParams)
        {

            uint idx = 0;

            List<Tuple<IMap, IMem>> readBack = new List<Tuple<IMap, IMem>>();

            foreach (object o in additionalParams)
            {
                if (o.GetType().IsPrimitive)
                {
                    // set parameter
                    err = Cl.SetKernelArg(kernel, idx, new IntPtr(System.Runtime.InteropServices.Marshal.SizeOf(o)), o);
                    assert(err, string.Format("error on idx {0}", idx));
                    ++idx;
                }
                else
                {
                    IBufParam p = o as IBufParam;
                    if (p != null)
                    {
                        IMap map = p.Buf as IMap;
                        if (map != null)
                        {

                            MemFlags memFlags = MemFlags.None;

                            switch (p.Scope & (BufParamScope.InOut))
                            {
                                case BufParamScope.In:
                                    memFlags = MemFlags.ReadOnly;
                                    break;
                                case BufParamScope.Out:
                                    memFlags = MemFlags.WriteOnly;
                                    break;
                                case BufParamScope.InOut:
                                    memFlags = MemFlags.ReadWrite;
                                    break;
                            }

                            // buffer created here regardless of purpose (in/out)
                            IMem cl_buf = Cl.CreateBuffer(_context, memFlags, map.ByteSize, out err);

                            if ((p.Scope & BufParamScope.In) != 0)
                            {
                                // enqueue write or read, set flags as readonly, writeonly or both, depending on Flags
                                map.EnqueueWrite(_commandsQueue, cl_buf);
                            }

                            if ((p.Scope & BufParamScope.Out) != 0)
                            {
                                readBack.Add(Tuple.Create(map, cl_buf));
                            }

                            // set pointer as parameter here regardless of purpose (in/out)
                            err = Cl.SetKernelArg(kernel, idx, intPtrSize, cl_buf);
                            assert(err, string.Format("error on idx {0}", idx));
                            ++idx;
                        }
                    }
                    else
                    {
                        IMem mem = o as IMem;
                        if (mem != null)
                        {
                            err = Cl.SetKernelArg(kernel, idx, intPtrSize, mem);
                            assert(err, string.Format("error on idx {0}", idx));
                            ++idx;
                        }
                    }
                }
            }

            IntPtr[] offsA = new IntPtr[offs.Length];
            for (int i = 0; i < offs.Length; ++i) { offsA[i] = new IntPtr(offs[i]); }
            IntPtr[] rangeA = new IntPtr[range.Length];
            for (int i = 0; i < range.Length; ++i) { rangeA[i] = new IntPtr(range[i]); }

            Event clevent;
            // execute
            err = Cl.EnqueueNDRangeKernel(_commandsQueue, kernel, 2, offsA, rangeA, null, 0, null, out clevent);
            clevent.Dispose();
            assert(err, "Cl.EnqueueNDRangeKernel");

            // read from output memory object into buffers for IMaps marked as output
            foreach (var r in readBack)
            {
                r.Item1.EnqueueRead(_commandsQueue, r.Item2);
            }

            // TODO: handle buffer to be released
        }

        public Map<float> gray(Map<byte> fm)
        {
            Map<float> result = new Map<float>(fm.W, fm.H);
            invokeKernel(_kernels["colorConv"], new int[] { 0, 0 }, new int[] { fm.W, fm.H }, fm.AsBufParam(BufParamScope.In), result.AsBufParam(BufParamScope.Out), fm.Stride, result.Stride);
            return result;
        }

        public Map<float> ScaleDownH(Map<float> fm)
        {
            Map<float> result = new Map<float>(fm.W / 2, fm.H);
            invokeKernel(_kernels["scaleDownH"], new int[] { 0, 0 }, new int[] { fm.W / 2, fm.H }, fm.AsBufParam(BufParamScope.In), result.AsBufParam(BufParamScope.Out), fm.W, fm.W / 2);
            return result;
        }

        public void blit(Map<float> src, Map<float> dst, int x, int y)
        {
            int w = Math.Min(src.W, dst.W - x);
            int h = Math.Min(src.H, dst.H - y);
            if ((w<=0)||(h<=0)) return;

            int srcIdx = 0;
            int dstIdx = x;    

            for (int yi = 0; yi < h; ++yi )
            {
                Array.Copy(src.Buf, srcIdx, dst.Buf, dstIdx, w);
                srcIdx += src.Stride; // srcIdx = yi * src.Stride
                dstIdx += dst.Stride; // dstIdx = yi * dst.Stride
            }
        }

        public int intMultSup(int a, int b)
        {
            return (((a - 1) / b) + 1) * b;
        }

        public int doPrism(Map<float> src, out Map<float> dst)
        {
            int w = intMultSup(src.W, 8);
            int h = src.H;
            dst = new Map<float>(w * 4, h); //  4=    | fractions  |   original  |            double            |

            blit(src, dst, w, 0);

            ErrorCode err;
            IMem clBuf = Cl.CreateBuffer(_context, MemFlags.ReadWrite, dst.ByteSize, out err);
            dst.EnqueueWrite(_commandsQueue, clBuf);

            invokeKernelNoSync(_kernels["scaleUpLinHPrism"], new int[] { 0, 0 }, new int[] { w, h }, clBuf, dst.Stride, w, w);
            invokeKernelNoSync(_kernels["scaleDownHPrism"], new int[] { 0, 0 }, new int[] { w, h }, clBuf, dst.Stride, w, -w / 2);
            invokeKernelNoSync(_kernels["scaleDownHPrism"], new int[] { 0, 0 }, new int[] { w, h }, clBuf, dst.Stride, w / 2, -w / 4);
            invokeKernelNoSync(_kernels["scaleDownHPrism"], new int[] { 0, 0 }, new int[] { w, h }, clBuf, dst.Stride, w / 4, -w / 8);

            dst.EnqueueRead(_commandsQueue, clBuf);

            return w;
        }

        public void doDisp(Map<float> srcL, Map<float> srcR, out Map<uint> disp)
        {
            int w = srcL.W / 4; // original image w = prismW/4
            int h = srcL.H;

            int offs = w; // original image starts at prismW/4

            disp = new Map<uint>(srcL.W, srcL.H, 1); // create a disp map big as the whole prism, 1 component (considering 2x ushort = 1 uint)
            disp.memset(UInt32.MaxValue);


            invokeKernelNoSync(_kernels["blockMatch3x3W"], new int[] { 1, 1, 1 }, new int[] { w-1, w/2-2, h },
                srcL.AsBufParam(BufParamScope.In), 
                offs,
                srcL.Stride,
                
                srcR.AsBufParam(BufParamScope.In),
                offs,
                srcR.Stride,
                
                disp.AsBufParam(BufParamScope.InOut),
                offs,
                disp.Stride, 
                
                w-1);
        }

        //public Map<float> ScaleUp(Map<float> fm)
        //{
        //    var tmp = invokeMapKernel<float, float>(_kernels["scaleUpV"], fm, fm.W, fm.H * 2, fm.W, fm.H, 0, 0);
        //    return invokeMapKernel<float, float>(_kernels["scaleUpH"], tmp, tmp.W * 2, tmp.H, tmp.W, tmp.H, 0, 0);
        //}

        //public Map<float> ScaleUpH(Map<float> fm)
        //{
        //    return invokeMapKernel<float, float>(_kernels["scaleUpH"], fm, fm.W * 2, fm.H, fm.W, fm.H, 0, 0);
        //}
    }
}
