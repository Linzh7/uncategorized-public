img=double(rgb2gray(imread('img.jpg')));

subplot(321);imshow(img,[]);

s=size(img);
w=0.2*2*pi;N=2*pi/w;
noise=10*ones(s(1),1)*sin(w*[1:s(2)]);
noisedImg=img+noise;
subplot(322);imshow(noisedImg,[]);
% imwrite(noisedImg,'noisedImg.jpg');

rawF=fftshift(fft2(img));
subplot(323);imshow(log(abs(rawF)),[]);
noisedF=fftshift(fft2(noisedImg));
subplot(324);imshow(log(abs(noisedF)),[]);

H=ones(s);
x0=s(1)/2+1;y0=s(2)/2+1;
x=x0;y=y0-round(s(2)/N);
width=10;
H(x,y-width:y+width)=0;
H(x,2*y0-y-width:2*y0-y+width)=0;



% I=ifftshift(filter2(H,noisedImg));
I=ifftshift(noisedF.*H);
imgRecover=ifft2(I);
% subplot(325);imshow(I,[]);
subplot(326);imshow(imgRecover,[]);
% recoverF=fftshift(fft2(imgRecover));
subplot(325);imshow(H,[]);