#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/pgm.h"

// Configuration for the Hough Transform
const int degreeInc = 2; // Increment in degrees for theta
const int degreeBins = 180 / degreeInc; // Number of theta bins
const int rBins = 100; // Number of bins for r
const double radInc = degreeInc * M_PI / 180; // Increment in radians for theta
//*****************************************************************
/*
Funcion para calcular el threshold
cpuht: puntero al acumulador
*/
int thresholdCalculus(int *cpuht){
  // Cálculo del umbral (ejemplo: promedio + 2 desviaciones estándar)
  const int degreeBins = 180 / degreeInc; // Número de bins en la dimensión theta
  const int rBins = 100; // Número de bins en la dimensión r
  int sum = 0, count = 0;
  for (int i = 0; i < degreeBins * rBins; i++) { // moverse por la matriz acumuladora
      sum += cpuht[i]; // sumar los valores de la matriz acumuladora
      if (cpuht[i] > 0) count++; // contar los valores no nulos
  }
  // Verificar si count es cero para evitar divisiones por cero
  if (count == 0) {
      std::cerr << "Error: No hay elementos no nulos en la matriz acumuladora." << std::endl;
      return -1; // Terminar o usar otro método de manejo de error
  }

  float avg = static_cast<float>(sum) / count; // Calcular el promedio
  float stddev = 0; // Inicializar la desviación estándar
  for (int i = 0; i < degreeBins * rBins; i++) { // moverse por la matriz acumuladora
      if (cpuht[i] > 0)  { // si el valor es no nulo
          stddev += pow(cpuht[i] - avg, 2); // sumar el cuadrado de la diferencia con el promedio
      }
  }
  stddev = sqrt(stddev / count); // Calcular la raíz cuadrada de la suma de los cuadrados de las diferencias con el promedio (desviacion estandar)

  // Cálculo del umbral
  int threshold = static_cast<int>(avg + 2 * stddev);
  return threshold;
}
// The CPU function returns a pointer to the accummulator
/*
Funcion para calcular la transformada de Hough en CPU
pic: imagen en escala de grises
w: ancho de la imagen
h: alto de la imagen
acc: acumulador
*/
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2; // El centro en x
  int yCent = h / 2; // El centro en y
  float rScale = 2 * rMax / rBins; // Escala del radio

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent; // Coordenada x respecto al centro de la imagen
            int yCoord = yCent - j;  // Coordenada y respecto al centro de la imagen
            float theta = 0;         // angulo actual
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta); // Calcular el radio de ese angulo
                int rIdx = (r + rMax) / rScale; // El radio se convierte a un indice discreto
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta (en el acumulador o los pesos)
                theta += radInc; // Incrementar el angulo a analizar
              }
          }
      }
}
//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ double d_Cos[degreeBins];
__constant__ double d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
/*
Funcion para calcular la transformada de Hough en GPU con memoria compartida
pic: imagen en escala de grises
w: ancho de la imagen
h: alto de la imagen
acc: acumulador
rMax: radio maximo
rScale: escala del radio
*/
__global__ void GPU_HoughTran_Shared(unsigned char *pic, int w, int h, int *acc, double rMax, double rScale) 
{
  //TODO
    extern __shared__ int localAcc[]; // Acumulador local en memoria compartida
    int gloID = blockIdx.x * blockDim.x + threadIdx.x; // Identificador global del hilo
    int tIdx = threadIdx.x; // Identificador local del hilo
    if (tIdx < degreeBins) { // Inicializar el acumulador local
        for (int rIdx = 0; rIdx < rBins; rIdx++) {
            localAcc[rIdx * degreeBins + tIdx] = 0;
        }
    }
    __syncthreads(); // Sincronizar los hilos del bloque
    int xCent = w / 2; // El centro en x
    int yCent = h / 2; // El centro en y
    if (gloID < w * h && pic[gloID] > 0) {
        int xCoord = gloID % w - xCent;
        int yCoord = yCent - gloID / w;

        // Iterar sobre todos los ángulos de theta
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (int)((r + rMax) / rScale + 0.5);
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1); // Usar operación atómica para acumulador
            }
        }
    }
    __syncthreads(); // Sincronizar los hilos del bloque
    // Transferir los datos del acumulador local al acumulador global en memoria compartida
    if (tIdx < degreeBins) {
        for (int rIdx = 0; rIdx < rBins; rIdx++) {
            atomicAdd(&acc[rIdx * degreeBins + tIdx], localAcc[rIdx * degreeBins + tIdx]);
        }
    }
}
//TODO Kernel memoria Constante
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale){
    // Global thread index
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    // Calculate relative coordinates to the center
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Process only edge pixels
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (int)((r + rMax) / rScale + 0.5);
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[rIdx * degreeBins + tIdx], 1); // Atomic update of accumulator
            }
        }
    }
}
// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
/*
Funcion para calcular la transformada de Hough en GPU memoria global
pic: imagen en escala de grises
w: ancho de la imagen
h: alto de la imagen
acc: acumulador
rMax: radio maximo
rScale: escala del radio
d_Cos: senos calculados
d_Sin: cosenos calculados
*/
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x; // Identificador global del hilo
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2; // El centro en x
  int yCent = h / 2; // El centro en y

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent; // Coordenada x respecto al centro de la imagen
  int yCoord = yCent - gloID / w; // Coordenada y respecto al centro de la imagen

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0) // Si el pixel esta coloreado (parte de una linea o un borde detectado) se procesa
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++) // 
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1); // Es para el acumulador que es compartido se le suma 1
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************
int main(int argc, char **argv) 
{
    int i;

    PGMImage inImg (argv[1]);

    int *cpuht; // Es la direccion del puntero donde estara el acumulador
    int w = inImg.x_dim; // Ancho de la imagen
    int h = inImg.y_dim; // Largo de la imagen

    auto startCPU = std::chrono::high_resolution_clock::now(); // Empezar a tomar tiempo de ejecucion CPU
    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht); // Llamar a funcion para calcular el acumulador
    auto endCPU = std::chrono::high_resolution_clock::now(); // Terminar de tomar tiempo de ejecucion CPU
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU; // Calcular tiempo de ejecucion CPU

    // pre-compute values to be stored
    double *pcCos = (double *)malloc(sizeof(double) * degreeBins);
    double *pcSin = (double *)malloc(sizeof(double) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
      pcCos[i] = cos (rad);
      pcSin[i] = sin (rad);
      rad += radInc;
    }

    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2; // Radio maximo
    float rScale = 2 * rMax / rBins; // Escala del radio
    // TODO eventualmente volver memoria global
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(double) * degreeBins); // Copiar valores precalculados de cosenos a la GPU en memoria constante
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(double) * degreeBins); // Copiar valores precalculados de senos a la GPU en memoria constante

    // setup and copy data from host to device
    unsigned char *d_in, *h_in; // Punteros a la imagen en la GPU (d_in)
    int *d_hough, *h_hough; // Punteros al acumulador en la GPU (d_hough)

    h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

    h_hough = (int *) malloc (degreeBins * rBins * sizeof (int)); 

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    //1 thread por pixel
    int blockNum = ceil (w * h / 256);
    size_t sharedMemSize = degreeBins * rBins * sizeof(int); // Tamaño de memoria compartida: sharedMemSize calcula cuánto espacio será necesario en la memoria compartida por bloque para el acumulador local
    // Cuda events to measure time
    cudaEvent_t start, stop; // Crear eventos para medir tiempo
    cudaEventCreate(&start); // Crear evento de inicio
    cudaEventCreate(&stop); // Crear evento de fin
    cudaEventRecord(start); // Empezar a medir tiempo
    // GPU calculation
    GPU_HoughTran_Shared <<< blockNum, 256, sharedMemSize >>> (d_in, w, h, d_hough, rMax, rScale); // Llamar al kernel de la GPU y ejecutarlo para calculo de acumulador
    cudaEventRecord(stop); // Terminar de medir tiempo
    // get results from device
    cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost); // Copiar el acumulador de la GPU a la CPU
    
    cudaEventSynchronize(stop); // Sincronizar eventos
    float milliseconds = 0; // Variable para almacenar tiempo
    cudaEventElapsedTime(&milliseconds, start, stop); // Calcular tiempo
    // compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
      if (cpuht[i] != h_hough[i])
        printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");

    printf("GPU Time: %f ms\nCPU Time: %f ms\n", milliseconds, elapsedCPU.count() * 1000);

    int threshold = thresholdCalculus(h_hough); // Calcular umbral

    // Dibujar la imagen
    drawImage("houghSharedCPU.jpg", inImg.pixels, w, h, threshold, cpuht, rScale, rMax); // Dibujar imagen con el acumulador de la CPU
    drawImage("houghSharedGPU.jpg", inImg.pixels, w, h, threshold, h_hough, rScale, rMax); // Dibujar imagen con el acumulador de la GPU

    // Liberación de memoria
    free(cpuht);
    free(pcCos);
    free(pcSin);
    free(h_hough);
    // Liberar memoria de la GPU
    cudaFree(d_in);
    cudaFree(d_hough);

    return 0;
}