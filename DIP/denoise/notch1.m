img=double(rgb2gray(imread('img.jpg')));

% subplot(321);imshow(img,[]);

s=size(img);
w=0.2*2*pi;N=2*pi/w;
noise=10*ones(s(1),1)*sin(w*[1:s(2)]);
src=img+noise;

% src = im2double(imread('noisedImg.jpg'));
% src = rgb2gray(src);
subplot(221);
imshow(src,[]);
title('原始图像');
[w h] = size(src);

srcf = fft2(src);
srcf = fftshift(srcf);
subplot(222);
imshow(log((abs(srcf))),[]);
%  低通滤波
% flt = zeros(size(src));
% rx1 = w/2;
% ry1 = h/2;
% r = min(w,h)/3;
% for i = 1:w
%     for j = 1:h
%         if(rx1-i)^2 +(ry1 - j)^2 <= r*r
%             flt(i,j) = 1;
%         end
%     end
% end
% 陷波滤波
flt = ones(size(src));
r = 10;
rx1 = w/2;
ry1 = h/2-round(s(2)/N);
for i = 1:w
    for j = 1:h
        if(rx1-i)^2 +(ry1 - j)^2 <= r*r
            flt(i,j) = 0;
        end
        if(w-rx1-i)^2 +(h-ry1 - j)^2 <= r*r
            flt(i,j) = 0;
        end
    end
end
subplot(223);
imshow(flt,[]);
title('滤波器图像');
dfimg = srcf.*flt;
dfimg = ifftshift(dfimg);
dimg = ifft2(dfimg,'symmetric');
subplot(224);
imshow(dimg,[]);title('滤波后');