%%
%parta
clc;clear;close all;
Fs = 1e06; Fc=1e04;F0=1e04;Bw=1e03;
Pulse = 1e-03;
length_of_pulse = Pulse*Fs;
One=ones(1,length_of_pulse);
Zero=-ones(1,length_of_pulse);
x=(sign(rand(1,100)-.5)+1)/2;
figure()
stem(x, 'r.');
xlabel('t');
ylabel('input pulse');
title('input pulse time domine ');
grid on;grid minor;
[in1,in2]=Divide(x);
input1=PulseShaping(in1,One,Zero);
input2=PulseShaping(in2,One,Zero);
t=1/Fs:1/Fs:length(input1)*1/Fs;
figure;
subplot(2,1,1)
plot(t,input1);
xlabel('t');
ylabel('input pulse first part');
title('input pulse first part time domine ');
grid on;grid minor;
subplot(2,1,2)
plot(t,input2);
xlabel('t');
ylabel('input pulse second part');
title('input pulse second part time domine ');
grid on;grid minor;
X=AnalogMod(input1,input2,Fs,Fc);
figure;
subplot(2,1,1);
plot(t,X);
xlabel('t');
ylabel('dumolated signal in channle');
title('dumolated signal in channle time domine ');
grid on;grid minor;
xlim([0,0.005]);
output=Channel(X,Fs,F0,Bw);
subplot(2,1,2);
plot(t,output);
ylabel('dumolated signal in channle after channel');
title('dumolated signal in channle after channel time domine ');
grid on;grid minor;
xlim([0,0.005]);
[out1,out2]=AnalogDemod(output,Fs,Bw,Fc);
figure;
subplot(2,1,1);
plot(t,out1);
xlabel('t');
ylabel('demolated signal part one ');
title('demolated signal part one time domine ');
grid on;grid minor;
ylim([-1.5,1.5]);
subplot(2,1,2);
plot(t,out2);
xlabel('t');
ylabel('demolated signal part two ');
title('demolated signal part two time domine ');
grid on;grid minor;
ylim([-1.5,1.5]);
[one_cor,zero_cor,out1]=MatchedFilt(real(out1),One,Zero);
figure;
subplot(2,1,1);
stem(one_cor,'r*')
ylabel('crrolation ');
title('crrolation of first part of signal with "one"  ');
grid on;grid minor;
subplot(2,1,2);
stem(zero_cor,'c.');
ylabel('crrolation ');
title('crrolation of first part of signal with "zero"  ');
grid on;grid minor;
 
[one_cor,zero_cor,out2]=MatchedFilt(real(out2),One,Zero);
figure;
subplot(2,1,1);
stem(one_cor,'r*')
ylabel('crrolation ');
title('crrolation of second  part of signal with "one"  ');
grid on;grid minor;
subplot(2,1,2);
stem(zero_cor,'c.');
ylabel('crrolation ');
title('crrolation of second part of signal with "zero"  ');
grid on;grid minor;
figure;
subplot(2,1,1)
stem(out1,'r*');
xlabel('t');
ylabel('input pulse first part AFTER CHANNLE');
title('input pulse first part AFTER CHANNLE time domine ');
grid on;grid minor;
subplot(2,1,2)
stem(out2,'g*');
xlabel('t');
ylabel('input pulse second part AFTER CHANNLE');
title('input pulse second part AFTER CHANNLE time domine ');
grid on;grid minor;
demulated=Combine(out1,out2);
figure();
subplot(2,1,1);
stem(demulated,'r*');
ylabel('input pulse after channle');
title('input pulse after channle time domine ');
grid on;grid minor;
subplot(2,1,2);
stem(x,'g*');
ylabel('input pulse');
title('input pulse time domine ');
grid on;grid minor;
%%
%partb
clc; clear;
Fs = 1e06;Fc=1e04;F0=1e04;Bw=1e03;
Pulse=1e-03;
length_of_pulse=Pulse*Fs;
One=ones(1,length_of_pulse);
zero=-ones(1,length_of_pulse);
snr=linspace(0,-800,800);
for i=1:1:length(snr)
    i
    x=(sign(rand(1,50)-.5)+1)/2;
    [in1,in2]=Divide(x);
    input1=PulseShaping(in1,One,zero);
    input2=PulseShaping(in2,One,zero);
    X = AnalogMod(input1,input2,Fs,Fc);
    X = awgn(X,snr(i));
    output = Channel(X,Fs,F0,Bw);
    [out1,out2] = AnalogDemod(output,Fs,Bw,Fc);
    [cor1,cor0,out1] = MatchedFilt(real(out1),One,zero);
    [cor1,cor0,out2] = MatchedFilt(real(out2),One,zero);
    demulated = Combine(out1,out2);
    er(i)=sum(abs(x-demulated))/length(x)*100;
end

figure;
plot(snr,er,'r');
xlabel('snr');
ylabel('erobability of error');
title('probability of error in snr domin');
grid on;
grid minor;
%%
%histogram
clc; clear;
Fs = 1e06;Fc=1e04;F0=1e04;Bw=1e03;
dt = 1/Fs;
Pulse=1e-03;
length_of_pulse=Pulse*Fs;
One=ones(1,length_of_pulse);
zero=-ones(1,length_of_pulse);
snr=linspace(0,-800,16);
figure;
for i=1:1:length(snr)
    i
    x=(sign(rand(1,50)-.5)+1)/2;
    [in1,in2]=Divide(x);
    input1=PulseShaping(in1,One,zero);
    input2=PulseShaping(in2,One,zero);
    X = AnalogMod(input1,input2,Fs,Fc);
    X = awgn(X,snr(i));
    output = Channel(X,Fs,F0,Bw);
    [out1,out2] = AnalogDemod(output,Fs,Bw,Fc);
    [cor1_1,cor0_1,out1] = MatchedFilt(real(out1),One,zero);
    [cor1_2,cor0_2,out2] = MatchedFilt(real(out2),One,zero);
    subplot(4,4,i);
    hist3([cor1_1',cor0_1'],'CDataMode','auto','FaceColor','interp');
    xlabel('cor1')
    ylabel('cor0')
    title(['snr of noise is' ,num2str(snr(i)), 'db'])
    grid on; grid minor;
    demulated = Combine(out1,out2);
    er(i)=sum(abs(x-demulated))/length(x)*100;
end

%%
%partd______500hz
clc;clear;close all;
Fs = 1e06; Fc=1e04;F0=1e04;Bw=1e03;
Pulse = 1e-03;
length_of_pulse = Pulse*Fs;
for i =1:length_of_pulse
    One(i)=sin(2*pi*500*i/Fs);
    Zero(i)=-sin(2*pi*500*i/Fs);
end
x=(sign(rand(1,100)-.5)+1)/2;
figure()
stem(x, 'r.');
xlabel('t');
ylabel('input pulse');
title('input pulse time domine ');
grid on;grid minor;
[in1,in2]=Divide(x);
input1=PulseShaping(in1,One,Zero);
input2=PulseShaping(in2,One,Zero);
t=1/Fs:1/Fs:length(input1)*1/Fs;
figure;
subplot(2,1,1)
plot(t,input1);
xlabel('t');
ylabel('input pulse first part');
title('input pulse first part time domine ');
grid on;grid minor;
subplot(2,1,2)
plot(t,input2);
xlabel('t');
ylabel('input pulse second part');
title('input pulse second part time domine ');
grid on;grid minor;
X=AnalogMod(input1,input2,Fs,Fc);
figure;
subplot(2,1,1);
plot(t,X);
xlabel('t');
ylabel('dumolated signal in channle');
title('dumolated signal in channle time domine ');
grid on;grid minor;
xlim([0,0.005]);
output=Channel(X,Fs,F0,Bw);
subplot(2,1,2);
plot(t,output);
ylabel('dumolated signal in channle after channel');
title('dumolated signal in channle after channel time domine ');
grid on;grid minor;
xlim([0,0.005]);
[out1,out2]=AnalogDemod(output,Fs,Bw,Fc);
figure;
subplot(2,1,1);
plot(t,out1);
xlabel('t');
ylabel('demolated signal part one ');
title('demolated signal part one time domine ');
grid on;grid minor;
ylim([-1.5,1.5]);
subplot(2,1,2);
plot(t,out2);
xlabel('t');
ylabel('demolated signal part two ');
title('demolated signal part two time domine ');
grid on;grid minor;
ylim([-1.5,1.5]);
[one_cor,zero_cor,out1]=MatchedFilt(real(out1),One,Zero);
figure;
subplot(2,1,1);
stem(one_cor,'r*')
ylabel('crrolation ');
title('crrolation of first part of signal with "one"  ');
grid on;grid minor;
subplot(2,1,2);
stem(zero_cor,'c.');
ylabel('crrolation ');
title('crrolation of first part of signal with "zero"  ');
grid on;grid minor;
 
[one_cor,zero_cor,out2]=MatchedFilt(real(out2),One,Zero);
figure;
subplot(2,1,1);
stem(one_cor,'r*')
ylabel('crrolation ');
title('crrolation of second  part of signal with "one"  ');
grid on;grid minor;
subplot(2,1,2);
stem(zero_cor,'c.');
ylabel('crrolation ');
title('crrolation of second part of signal with "zero"  ');
grid on;grid minor;
figure;
subplot(2,1,1)
stem(out1,'r*');
xlabel('t');
ylabel('input pulse first part AFTER CHANNLE');
title('input pulse first part AFTER CHANNLE time domine ');
grid on;grid minor;
subplot(2,1,2)
stem(out2,'g*');
xlabel('t');
ylabel('input pulse second part AFTER CHANNLE');
title('input pulse second part AFTER CHANNLE time domine ');
grid on;grid minor;
demulated=Combine(out1,out2);
figure();
subplot(2,1,1);
stem(demulated,'r*');
ylabel('input pulse after channle');
title('input pulse after channle time domine ');
grid on;grid minor;
subplot(2,1,2);
stem(x,'g*');
ylabel('input pulse');
title('input pulse time domine ');
grid on;grid minor;

%%
%%noise 500hz
clc; clear;
Fs = 1e06;Fc=1e04;F0=1e04;Bw=1e03;
dt = 1/Fs;
Pulse=1e-03;
length_of_pulse=Pulse*Fs;
for i =1:length_of_pulse
    One(i)=sin(2*pi*500*i/Fs);
    zero(i)=-sin(2*pi*500*i/Fs);
end
    
snr=linspace(0,-800,25);
figure;
for i=1:1:length(snr)
    i
    x=(sign(rand(1,50)-.5)+1)/2;
    [in1,in2]=Divide(x);
    input1=PulseShaping(in1,One,zero);
    input2=PulseShaping(in2,One,zero);
    X = AnalogMod(input1,input2,Fs,Fc);
    X = awgn(X,snr(i));
    output = Channel(X,Fs,F0,Bw);
    [out1,out2] = AnalogDemod(output,Fs,Bw,Fc);
    [cor1,cor0,out1] = MatchedFilt(real(out1),One,zero);
    [cor1,cor0,out2] = MatchedFilt(real(out2),One,zero);
    demulated = Combine(out1,out2);
    er(i)=sum(abs(x-demulated))/length(x)*100;
end

figure;
plot(snr,er,'r');
xlabel('snr');
ylabel('erobability of error');
title('probability of error in snr domin');
grid on;
grid minor;
%%
%histogram 500 hz
clc; clear;
Fs = 1e06;Fc=1e04;F0=1e04;Bw=1e03;
dt = 1/Fs;
Pulse=1e-03;
length_of_pulse=Pulse*Fs;
for i =1:length_of_pulse
    One(i)=sin(2*pi*500*i/Fs);
    zero(i)=-sin(2*pi*500*i/Fs);
end
snr=linspace(0,-800,16);
figure;
for i=1:1:length(snr)
    i
    x=(sign(rand(1,50)-.5)+1)/2;
    [in1,in2]=Divide(x);
    input1=PulseShaping(in1,One,zero);
    input2=PulseShaping(in2,One,zero);
    X = AnalogMod(input1,input2,Fs,Fc);
    X = awgn(X,snr(i));
    output = Channel(X,Fs,F0,Bw);
    [out1,out2] = AnalogDemod(output,Fs,Bw,Fc);
    [cor1_1,cor0_1,out1] = MatchedFilt(real(out1),One,zero);
    [cor1_2,cor0_2,out2] = MatchedFilt(real(out2),One,zero);
    subplot(4,4,i);
    hist3([cor1_1',cor0_1'],'CDataMode','auto','FaceColor','interp');
    xlabel('cor1')
    ylabel('cor0')
    title(['snr of noise is' ,num2str(snr(i)), 'db'])
    grid on; grid minor;
    demulated = Combine(out1,out2);
    er(i)=sum(abs(x-demulated))/length(x)*100;
end

%%
%bakhsh se
clc; clear;
Fs = 1e06;Fc=1e04;F0=1e04;Bw=1e03;
Pulse=1e-03;
length_of_pulse=Pulse*Fs;
One=ones(1,length_of_pulse);
zero=-ones(1,length_of_pulse);
snr=linspace(-700,-800,5);
for i=1:1:length(snr)
    i
    snr(i);
    x = floor((rand(1,100)*256));
    a = SourceGenerator(x);
    [in1,in2]=Divide(a);
    input1=PulseShaping(in1,One,zero);
    input2=PulseShaping(in2,One,zero);
    X = AnalogMod(input1,input2,Fs,Fc);
    X = awgn(X,snr(i));
    output = Channel(X,Fs,F0,Bw);
    [out1,out2] = AnalogDemod(output,Fs,Bw,Fc);
    [cor1,cor0,out1] = MatchedFilt(real(out1),One,zero);
    [cor1,cor0,out2] = MatchedFilt(real(out2),One,zero);
    demulated = Combine(out1,out2);
    a = OutputDecoder(demulated);
    er(i)=sum(abs((x-a)).^2)
end

figure;
plot(snr,er,'r');
xlabel('snr');
ylabel('square of error');
title('square of error in snr domin');
grid on;
grid minor;












%%
%functions
 function out = SourceGenerator(x)
 out = [];
    for i = 1:length(x)
        out = [out,de2bi(x(i),8)];
    end   
 end
 
function out = OutputDecoder(x)
 out = [];
    for i = 1:8:length(x)
        out = [out,bi2de([x(i),x(i+1),x(i+2),x(i+3),x(i+4),x(i+5),x(i+6),x(i+7)])];
    end   
 end
function [ a , b ] = Divide(x)
    for i = 1:length(x)
        if mod(i,2) == 0 
            b (i/2) = x(i);
        end
        if mod(i,2) == 1 
            a ((i+1)/2) = x(i);
        end
    end
        
end

function out = Combine(input1,input2)
out=[];
for i=1:1:length(input1)
    out = [out,input1(i),input2(i)];
end
end


function out = PulseShaping( input , one , zero)
    out = [];
    for i = 1:length(input)
        if input(i) == 1 ; 
            out = [out ,one];
        end
        
        if input(i) == 0 ; 
          out = [out ,zero];
        end
    end  
    end
        

function out = AnalogMod( input1 , input2 , Fs ,Fc)
    for i = 1: length(input1)
        out(i) = input1(i) * cos(  2*pi*Fc * i / Fs)...
            +input2(i) * sin(  2* pi* Fc * i / Fs);
    end
end

function out = Channel( input , Fs ,Fc ,Bw)
    num = length( input );
    fft_signal = fftshift (fft(input));
    F = -Fs/2 : Fs/num : Fs/2 - Fs/num;
    out_fft = zeros(1,length(fft_signal));
    min2 = floor((Fc-Bw/2+Fs/2)/(Fs/num));
    max2 = floor((Fc+Bw/2+Fs/2)/(Fs/num));
    min1 = floor((-Fc-Bw/2+Fs/2)/(Fs/num));
    max1 = floor(-Fc+Bw/2+Fs/2)/(Fs/num);
    for i = min1-100:1:max1+100
        if abs(F(i)-Fc)<=Bw/2
            
            out_fft(i) = fft_signal(i);
        
        elseif abs(F(i)+Fc)<=Bw/2
            out_fft(i) = fft_signal(i);
        else
            out_fft(i) = 0; 
        end
    end
    for i = min2-100:1:max2+100
        if abs(F(i)-Fc)<=Bw/2
            
            out_fft(i) = fft_signal(i);
        
        elseif abs(F(i)+Fc)<=Bw/2
            out_fft(i) = fft_signal(i);
        else
            out_fft(i) = 0; 
        end
    end
    out  = ifft(ifftshift(out_fft));
end

function [x1,x2]=AnalogDemod(input,Fs,Bw,Fc)
t=1/Fs:1/Fs:(length(input)*1/Fs);
Cos_vec = cos(2*pi*Fc*t);
Sin_vec = sin(2*pi*Fc*t);
x1=2*input.*Cos_vec;
x2=2*input.*Sin_vec ;
X1=fftshift(fft(x1));
X2=fftshift(fft(x2));
F=linspace(0,Fs,length(X1))-Fs/2;
min = floor(((Fs/2)-(Bw/2))/(Fs/length(X1)));
max = floor(((Fs/2)+(Bw/2))/(Fs/length(X1)));
out1 = zeros(1,length(X1));
out2 = zeros(1,length(X1));
for i=min-100:1:max+100
   if(abs(F(i))<Bw)
       out1(i)=X1(i);
       out2(i)=X2(i);
   end
end
x1=ifft(ifftshift(out1));
x2=ifft(ifftshift(out2));
end

function [cor_one,cor_zero,out]=MatchedFilt(input,one,zero)
psd_one = one.*one;
Energy=sum(psd_one);
cor_one=conv(input,one);
cor_zero=conv(input,zero);
out=zeros(1,length(input)/length(zero));
for i=1:length(out)
    if cor_one(i*length(zero))>=Energy/2
        out(i)=1;
    end
end
end