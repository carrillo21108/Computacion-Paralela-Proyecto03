# Computacion-Paralela-Proyecto03

## ¿Cómo levantar el contenedor?

### Obtener la imagen

```bash
docker pull diggspapu/parallel-mpi-mp-cuda
```

### Construir el contenedor

```bash
docker build -t paralela-proyecto-3 .
```

### Levantar el contenedor

```bash
docker run --gpus all -it --name=paralela-proyecto-3 -v $(pwd):/parallel_computing paralela-proyecto-3
```

### Levantar el contenedor cuando ya se creo el contenedor

```bash
docker start -i paralela-proyecto-3
```

### Descargar librería para generar imágenes

```bash
apt-get install cmake libopencv-dev
```

Poner cualquier configuración de regiones
