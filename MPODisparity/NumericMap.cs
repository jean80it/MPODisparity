using OpenCL.Net;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace MPODisparity
{
    [Flags]
    public enum BufParamScope 
    { 
        In = 1,
        Out = 2,
        InOut = 3
    }

    public interface IBufParam
    {
        object Buf { get; }
        //IMem MemBuf { get; }

        BufParamScope Scope { get; }
    }

    public interface IMap
    {
        int Size { get; }
        int W { get; }
        int H { get; }
        int Stride { get; }
        int ByteSize { get; }
        int LineStart(int y);
        
        void EnqueueWrite(CommandQueue commandsQueue, IMem inputMapBuffer);
        void EnqueueRead(CommandQueue commandsQueue, IMem mapBuffer);

        BufParam AsBufParam(BufParamScope flags);
    }

    public interface IMap<T> : IMap
        where T : struct
    {
        T[] Buf { get; }
        T this[int linearX]{get;}
    }

    public class Map<T> : IMap<T>
        where T:struct
    {
        public T[] Buf { get; protected set; }

        public int W { get; protected set; }
        public int H { get; protected set; }
        public int Stride { get; protected set; }
        public int Components { get; protected set; }

        public Map(int w, int h)
            : this(w, h, 1, w)
        { }

        public Map(int w, int h, int components)
            : this(w, h, components, w * components)
        { }

        public Map(int w, int h, int components, int stride)
        {
            W = w;
            H = h;
            Stride = stride;
            Components = components;
            Buf = new T[Stride * H];
        }

        public int LineStart(int y)
        {
            return y * Stride;
        }

        public T this[int linearX]
        {
            get { return Buf[linearX]; }
            set { Buf[linearX] = value; }
        }

        public T this[int x, int y]
        {
            get
            {
                return Buf[x * Components + (y * Stride)];
            }

            set
            {
                Buf[x * Components + (y * Stride)] = value;
            }
        }

        public int Size
        {
            get { return Buf.Length; }
        }

        public int ByteSize
        {
            get
            {
                return Size * System.Runtime.InteropServices.Marshal.SizeOf(typeof(T)); 
            }
        }

        public static Map<byte> FromBitmapRGB(Bitmap bmp)
        {
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            Map<byte> src = new Map<byte>(bd.Width, bd.Height, 3, bd.Stride);
            Marshal.Copy(bd.Scan0, src.Buf, 0, src.Size);
            bmp.UnlockBits(bd);
            return src;
        }

        public static Bitmap Map2Bmp(Map<float> imgf, float k)
        {
            int h = imgf.H;
            int w = imgf.W;
            int stride = imgf.Stride;

            var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);

            BitmapData dstData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            int pixelSize = 4;

            unsafe
            {
                var dstStride = dstData.Stride;
                byte* dstRow = (byte*)dstData.Scan0;
                int srcLineStart = 0;
                for (int y = 0; y < h; ++y)
                {
                    int srcIdx = srcLineStart;
                    int wb = w * pixelSize;
                    for (int x = 0; x < wb; x += pixelSize)
                    {
                        byte b = (byte)(imgf[srcIdx] * k);
                        dstRow[x] = b;
                        dstRow[x + 1] = b;
                        dstRow[x + 2] = b;
                        dstRow[x + 3] = 255;
                        ++srcIdx;
                    }
                    srcLineStart += stride;
                    dstRow += dstStride;
                }
            }

            bmp.UnlockBits(dstData);
            return bmp;
        }

        public static Bitmap Map2Bmp(Map<uint> imgf, uint mask, float k)
        {
            int h = imgf.H;
            int w = imgf.W;
            int stride = imgf.Stride;

            var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);

            BitmapData dstData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            int pixelSize = 4;

            unsafe
            {
                var dstStride = dstData.Stride;
                byte* dstRow = (byte*)dstData.Scan0;
                int srcLineStart = 0;
                for (int y = 0; y < h; ++y)
                {
                    int srcIdx = srcLineStart;
                    int wb = w * pixelSize;
                    for (int x = 0; x < wb; x += pixelSize)
                    {
                        byte b = (byte)((imgf[srcIdx] & mask) * k);
                        dstRow[x] = b;
                        dstRow[x + 1] = b;
                        dstRow[x + 2] = b;
                        dstRow[x + 3] = 255;
                        ++srcIdx;
                    }
                    srcLineStart += stride;
                    dstRow += dstStride;
                }
            }

            bmp.UnlockBits(dstData);
            return bmp;
        }

        public static Bitmap Map2Bmp(Map<byte> imgf)
        {
            int h = imgf.H;
            int w = imgf.W;
            int stride = imgf.Stride;

            var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);

            BitmapData dstData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            int pixelSize = 4;

            unsafe
            {
                var dstStride = dstData.Stride;
                byte* dstRow = (byte*)dstData.Scan0;
                int srcLineStart = 0;
                for (int y = 0; y < h; ++y)
                {
                    int srcIdx = srcLineStart;
                    int wb = w * pixelSize;
                    for (int x = 0; x < wb; x += pixelSize)
                    {
                        dstRow[x] = (byte)(imgf[srcIdx]);
                        ++srcIdx;
                        dstRow[x + 1] = (byte)(imgf[srcIdx]);
                        ++srcIdx;
                        dstRow[x + 2] = (byte)(imgf[srcIdx]);
                        ++srcIdx;
                        dstRow[x + 3] = 255;
                        //++srcIdx;
                    }
                    srcLineStart += stride;
                    dstRow += dstStride;
                }
            }

            bmp.UnlockBits(dstData);
            return bmp;
        }

        private void assert(ErrorCode err, string message)
        {
            if (err != ErrorCode.Success)
            {
                throw new Exception("error:" + err + "; " + message);
            }
        }

        public void EnqueueWrite(CommandQueue commandsQueue, IMem mapBuffer)
        {
            Event clevent;
            var err = Cl.EnqueueWriteBuffer(commandsQueue, mapBuffer, Bool.True, IntPtr.Zero, new IntPtr(ByteSize), (object)Buf, 0, null, out clevent);
            assert(err, "Cl.EnqueueWriteBuffer");
            clevent.Dispose();
        }

        public void EnqueueRead(CommandQueue commandsQueue, IMem mapBuffer)
        {
            Event clevent;
            var err = Cl.EnqueueReadBuffer(commandsQueue, mapBuffer, Bool.True, IntPtr.Zero, new IntPtr(ByteSize), (object)Buf, 0, null, out clevent);
            assert(err, "Cl.EnqueueReadBuffer");
            clevent.Dispose();
        }

        public BufParam AsBufParam(BufParamScope flags)
        {
            return BufParam.Create(this, flags);
        }

        internal void memset(T v)
        {
            for (int i = 0; i < this.Buf.Length; ++i)
            {
                this.Buf[i] = v;
            }
        }
    }

    public class BufParam : IBufParam
    {
        public BufParamScope Scope { get; private set; }
        public object Buf { get; private set; }
        //public IMem MemBuf { get; private set; }

        public static BufParam Create(object buf, BufParamScope scope)
        {
            return new BufParam() { Buf = buf, Scope = scope };
        }
    }
}