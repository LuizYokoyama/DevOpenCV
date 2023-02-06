package com.ls.DevOpenCV;

import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.OCRResult;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

@SpringBootApplication
public class DevOpenCvApplication {


	public static final String DATA_PATH = "/usr/share/tessdata/";
	public static final int ENGINE_MODE = 1;
	public static final int PAGE_MODE = 1;
	public static final String LANG = "por";



	public static void main(String[] args) {
		//SpringApplication.run(DevOpenCvApplication.class, args);
		opencv("/home/luiz/Downloads/test/c1.jpeg");
		computeSkew("/home/luiz/Downloads/test/grey.png");
		//doOcr("/home/luiz/Downloads/test/grey.png");
		getConfidence("/home/luiz/Downloads/test/c1.jpeg");


	}


	public static void opencv(String inFile){

		OpenCV.loadLocally();


		Mat original = Imgcodecs.imread(inFile);

		Mat gray = new Mat(original.rows(), original.cols(), original.type());
		Mat blur = new Mat(original.rows(), original.cols(), original.type());
		Mat unSharp = new Mat(original.rows(), original.cols(), original.type());
		Mat binary = new Mat(original.rows(), original.cols(), original.type());
		Mat detectedEdges = new Mat(original.rows(), original.cols(), original.type());
		MatOfInt params = new MatOfInt(Imgcodecs.IMWRITE_PNG_COMPRESSION);

		Imgproc.cvtColor(original, gray, Imgproc.COLOR_RGB2GRAY, 0);
		Imgcodecs.imwrite("/home/luiz/Downloads/test/grey.png", gray, params);

		Imgproc.GaussianBlur(gray, blur, new Size(3, 3), 3);
		Imgcodecs.imwrite("/home/luiz/Downloads/test/blur.png", blur, params);

		Core.addWeighted(blur, 0.5, blur, 0.5, 0, unSharp);
		Imgcodecs.imwrite("/home/luiz/Downloads/test/unSharp.png", unSharp, params);

		Imgproc.threshold(unSharp,binary,50,255, THRESH_BINARY);
		Imgcodecs.imwrite("/home/luiz/Downloads/test/binary.png", binary, params);

		Mat grad_x = new Mat();
		Mat grad_y = new Mat();
		Mat abs_grad_x = new Mat();
		Mat abs_grad_y = new Mat();
		int ddepth = CvType.CV_16S;
		// Gradient X
		Imgproc.Sobel(gray, grad_x, ddepth, 1, 0);
		Core.convertScaleAbs(grad_x, abs_grad_x);

		// Gradient Y
		Imgproc.Sobel(gray, grad_y, ddepth, 0, 1);
		Core.convertScaleAbs(grad_y, abs_grad_y);

		// Total Gradient (approximate)
		Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, detectedEdges);
		Imgcodecs.imwrite("/home/luiz/Downloads/test/detectedEdges.png", detectedEdges, params);

		//Imgcodecs.imwrite("/home/luiz/Downloads/cupom3.png", gray, params);


	}

	public static void getConfidence(String inFile){

		Mat mat = Imgcodecs.imread(inFile);
		BufferedImage bufferedImage = mat2Img(mat);
		Tesseract tesseract = new Tesseract();
		tesseract.setOcrEngineMode(ENGINE_MODE);
		tesseract.setPageSegMode(PAGE_MODE);
		tesseract.setLanguage(LANG);
		tesseract.setDatapath(DATA_PATH);

		List<ITesseract.RenderedFormat> renderedFormats = new ArrayList<>();
		renderedFormats.add(ITesseract.RenderedFormat.TEXT);
		OCRResult ocrResult;
		try {
			ocrResult = tesseract.createDocumentsWithResults(bufferedImage, "", "",renderedFormats, 1 );
		} catch (TesseractException e) {
			throw new RuntimeException(e);
		}

		System.out.println("Confidence: " + ocrResult.getConfidence());
		System.out.println(ocrResult.getWords().toString());

	}

	public static void doOcr(String inFile){

		Tesseract tesseract = new Tesseract();

		tesseract.setOcrEngineMode(ENGINE_MODE);
		tesseract.setPageSegMode(PAGE_MODE);
		tesseract.setLanguage(LANG);
		tesseract.setDatapath(DATA_PATH);


		/*byte[] imageByte = Base64.getDecoder().decode("");
		ByteArrayInputStream bis = new ByteArrayInputStream(imageByte);
		BufferedImage bufferedImage;
		try {
			bufferedImage = ImageIO.read(bis);
			bis.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
*/
		String text;
		File file = new File(inFile);
		try {
			text = tesseract.doOCR(file);
		} catch (TesseractException e) {
			throw new RuntimeException(e);
		}

		System.out.println(text);

	}

	public static Mat deskew(Mat src, double angle) {
		Point center = new Point(src.width()/2, src.height()/2);
		Mat rotImage = Imgproc.getRotationMatrix2D(center, angle, 1.0); //1.0 means 100 % scale
		Size size = new Size(src.width(), src.height());
		Imgproc.warpAffine(src, src, rotImage, size, Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
		return src;
	}

	public static void computeSkew( String inFile ) {
		//Load this image in grayscale
		Mat img = Imgcodecs.imread( inFile, Imgcodecs.IMREAD_GRAYSCALE );

		//Binarize it
		//Use adaptive threshold if necessary
		//Imgproc.adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);
		Imgproc.threshold( img, img, 200, 255, THRESH_BINARY );

		//Invert the colors (because objects are represented as white pixels, and the background is represented by black pixels)
		Core.bitwise_not( img, img );
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

		//We can now perform our erosion, we must declare our rectangle-shaped structuring element and call the erode function
		Imgproc.erode(img, img, element);

		//Find all white pixels
		Mat wLocMat = Mat.zeros(img.size(),img.type());
		Core.findNonZero(img, wLocMat);

		//Create an empty Mat and pass it to the function
		MatOfPoint matOfPoint = new MatOfPoint( wLocMat );

		//Translate MatOfPoint to MatOfPoint2f in order to user at a next step
		MatOfPoint2f mat2f = new MatOfPoint2f();
		matOfPoint.convertTo(mat2f, CvType.CV_32FC2);

		//Get rotated rect of white pixels
		RotatedRect rotatedRect = Imgproc.minAreaRect( mat2f );

		Point[] vertices = new Point[4];
		rotatedRect.points(vertices);
		List<MatOfPoint> boxContours = new ArrayList<>();
		boxContours.add(new MatOfPoint(vertices));
		Imgproc.drawContours( img, boxContours, 0, new Scalar(128, 128, 128), -1);

		double resultAngle = rotatedRect.angle;
		if (rotatedRect.size.width > rotatedRect.size.height)
		{
			//rotatedRect.angle += 90.f;
		}

		//Or
		//rotatedRect.angle = rotatedRect.angle < -45 ? rotatedRect.angle + 90.f : rotatedRect.angle;

		Mat result = deskew( Imgcodecs.imread( inFile ), -7 );
		Imgcodecs.imwrite( "/home/luiz/Downloads/test/deskewed2.jpg", result );

	}

	private static BufferedImage mat2Img(Mat m) {
		if (!m.empty()) {
			int type = BufferedImage.TYPE_BYTE_GRAY;
			if (m.channels() > 1) {
				type = BufferedImage.TYPE_3BYTE_BGR;
			}
			int bufferSize = m.channels() * m.cols() * m.rows();
			byte[] b = new byte[bufferSize];
			m.get(0, 0, b); // get all the pixels
			BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
			final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
			System.arraycopy(b, 0, targetPixels, 0, b.length);
			return image;
		}

		return null;
	}

	private static Mat img2Mat(BufferedImage in)
	{
		Mat out;
		byte[] data;
		int r, g, b;

		out = new Mat(in.getHeight(), in.getWidth(), CvType.CV_8UC3);
		data = new byte[in.getWidth() * in.getHeight() * (int)out.elemSize()];
		int[] dataBuff = in.getRGB(0, 0, in.getWidth(), in.getHeight(), null, 0, in.getWidth());
		for(int i = 0; i < dataBuff.length; i++)
		{
			data[i*3] = (byte) ((dataBuff[i] >> 16) & 0xFF);
			data[i*3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
			data[i*3 + 2] = (byte) ((dataBuff[i] >> 0) & 0xFF);
		}

		out.put(0, 0, data);
		return out;
	}




}
