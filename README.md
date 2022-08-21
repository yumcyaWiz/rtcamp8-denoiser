# rtcamp8-denoiser

[レイトレ合宿8](https://sites.google.com/view/raytracingcamp8/)のデノイズ部門用に作成したデノイザーです。

## Requirements

* CUDA

## Build

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

## Run

```
./main <filepath-to-color.hdr> <filepath-to-albedo.hdr> <filepath-to-normal.hdr>
```