#include <opencv2\opencv.hpp>

using namespace cv;

/// Global variables
Mat src, dst,mask;
int morph_elem = 0;
int morph_size = 0;
int const max_kernel_size = 21;

char* window_name = "Transformaciones Morfologicas";

void Morphology_Operations(int, void*); //Encabezado de las funciones

int main(int argc, char** argv) //Funcion Main
{
    src = imread("C:/Users/Paco/Dropbox/Vision Artificial/FINAL/1_res.png"); //Abrir una imagen
   
    cvtColor(src, src, COLOR_RGB2GRAY);
    if (!src.data)
    {
        return -1;
    }
    namedWindow(window_name, 0);

    //Creacion de barra de desplazamiento para seleccionar el tamaño de kernel
    createTrackbar("Tamaño", window_name, &morph_size, max_kernel_size, Morphology_Operations);

    Morphology_Operations(0, 0);//Llamada a la funcion

    waitKey(0);
    return 0;
}

void Morphology_Operations(int, void*) //Funcion para las operaciones morfologicas
{
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    Mat roi(src.rows, src.cols, src.type(), cv::Scalar::all(0));
    //Aplicar la operacion seleccionada
    morphologyEx(src, dst, MORPH_OPEN, element);
    //Aplicar Threshold
    threshold(dst, mask, 90, 255, 0);
    imshow(window_name, dst);
    imshow("Thresh", mask);
    src.copyTo(roi, mask);
    imshow("ROI", roi);
    imwrite("C:/Users/Paco/Dropbox/Vision Artificial/FINAL/1roi.jpg", roi);
}
