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

using OpenCV;

namespace MPODisparity
{
    class Program
    {
        static void Main(string[] args)
        {
            //string filename = args[0];
            string filename = @"E:\MPOs\P1200704.MPO";

            #region extract left and right JPEG from MPO
            UInt32 signature = 0;

            List<Image> images = new List<Image>(2);

            byte[] buf;
            int pos = 0, lastPos = 0, len;

            using (var fs = new FileStream(filename, FileMode.Open))
            {
                buf = new byte[len = (int)fs.Length];
                fs.Read(buf, 0, len);
            }

            while (pos < len)
            {
                do
                {
                    signature = (signature << 8) | buf[pos++];
                }
                while ((signature != 4292411361) && (pos < len));

                if (lastPos != 0)
                {
                    using (MemoryStream ms = new MemoryStream(buf, lastPos - 4, pos - lastPos + ((pos == len) ? 4 : 0)))
                    {
                        images.Add(Image.FromStream(ms));
                    }
                }
                lastPos = pos;
            }
            #endregion // extract left and right JPEG from MPO

            // DEBUG: save L and R images
            {
                (new Bitmap(images[0])).Save("L.png", ImageFormat.Png);
                (new Bitmap(images[1])).Save("R.png", ImageFormat.Png);
            }

            // create cl proxy
            CLStuff clHelper = new CLStuff();

            // get Luma only
            Map<float> lMap = clHelper.gray(Map<byte>.FromBitmapRGB(new Bitmap(images[0])));
            Map<float> rMap = clHelper.gray(Map<byte>.FromBitmapRGB(new Bitmap(images[1])));

            // compute prisms
            Map<float> lPrism;
            int w = clHelper.doPrism(lMap, out lPrism);
            Map<float> rPrism;
            clHelper.doPrism(rMap, out rPrism); // same w

            // execute kernel to compute match error: one execution per pixel per row, divided by w

            // e.g. first small version of image: enqueue search for first row checking against all remaining pixels in a line
            // so, if w=200 h=1000, work item size is 200x1000 for the first row, 199x1000 for the second, etc...
            // then, we pass to next bigger image etc..
            // we save computed error on a big buffer.

            Map<uint> disp;
            clHelper.doDisp(lPrism, rPrism, out disp);

            clHelper.finish();

            // DEBUG: save prisms 
            {
                (Map<float>.Map2Bmp(lPrism, 1)).Save("prismL.png", ImageFormat.Png);
                (Map<float>.Map2Bmp(rPrism, 1)).Save("prismR.png", ImageFormat.Png);
            }

            // save displacement
            (Map<float>.Map2Bmp(disp, 65535, (1.0f / 256.0f))).Save("disp.png", ImageFormat.Png);

            // save error
            (Map<float>.Map2Bmp(disp, ((uint)65535 << 16), (1.0f / 16777216.0f))).Save("dispErr.png", ImageFormat.Png);
        }
    }
}
