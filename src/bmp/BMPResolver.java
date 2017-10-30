package bmp;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class BMPResolver {
    private String filePath;
    BufferedImage image;
    int width;
    int height;
    int[][] imageFeature;

    public BMPResolver(int width, int height, String path) throws IOException {
        this.width = width;
        this.height = height;
        this.filePath = path;
        try {
            image = ImageIO.read(new java.io.File(filePath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (image == null) {
            System.out.println("path error");
            return;
        }

        imageFeature = new int[width][height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int tempFeature = image.getRGB(i, j);
                if ( tempFeature == -1) {
                    imageFeature[j][i] = '\0';
                } else {
                    imageFeature[j][i] = 1;
                }
            }
        }
//        for (int i = 0; i < height; i++) {
//            for (int j = 0; j < width; j++) {
//                System.out.print(imageFeature[i][j]);
//            }
//            System.out.println();
//        }
    }

    public double[] getInputVector() {
        double[] tempResult = new double[height * width];
        int index = 0;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                tempResult[index] = imageFeature[i][j];
                index++;
            }
        }
        return tempResult;
    }

}
