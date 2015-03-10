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
            UInt32 signature = 0;

            List<Image> images = new List<Image>(2);

            byte[] buf;
            int pos = 0, lastPos = 0, len;

            string filename = @"E:\MPOs\P1200704.MPO";

            using (var fs = new FileStream(filename, FileMode.Open))
            {
                buf = new byte[len = (int)fs.Length];
                fs.Read(buf, 0, len);
            }

            while(pos < len)
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

            CLStuff clHelper = new CLStuff();

            Map<float> lMap = clHelper.gray(Map<byte>.FromBitmapRGB(new Bitmap(images[0])));
            Map<float> rMap = clHelper.gray(Map<byte>.FromBitmapRGB(new Bitmap(images[1])));

            Map<float> lPrysm;
            int w = clHelper.doPrysm(lMap, out lPrysm);
            Map<float> rPrysm;
            clHelper.doPrysm(rMap, out rPrysm); // same w
            
            // execute kernel to compute match error: one execution per pixel per row, divided by w

            // e.g. first small version of image: enqueue search for first row checking against all remaining pixels in a line
            // so, if w=200 h=1000, work item size is 200x1000 for the first row, 199x1000 for the second, etc...
            // then, we pass to next bigger image etc..
            // we save computed error on a big buffer.


            //{
            //    Bitmap bmpOut = Map<float>.Map2Bmp(lPrysm, 1);
            //    bmpOut.Save("scalL.png", ImageFormat.Png);
            //}

            //{
            //    Bitmap bmpOut = Map<float>.Map2Bmp(rPrysm, 1);
            //    bmpOut.Save("scalR.png", ImageFormat.Png);
            //}
            
            (new Bitmap(images[0])).Save("L.png", ImageFormat.Png);
            (new Bitmap(images[1])).Save("R.png", ImageFormat.Png);

            OpenCV.Net.StereoBM s = new OpenCV.Net.StereoBM();
            //s.FindStereoCorrespondence()
        }
    }
}
