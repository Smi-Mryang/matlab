# matlab
计算、神经网络
clc;
clear all;
[xdata,textdata]=xlsread("exp12_4_2.xls");% （读取excel中的问当数据）
%读取货运量数据
[n,m]=size(xdata);
%输入样本数量为20
Train_Num=20;
%测试样本数量也为20
Test_Num=Train_Num;
%预测样本数量为6
Sim_Num=2;
%提取输入Input和输出数据Output数据
Input=[xdata(1:Train_Num,3) xdata(1:Train_Num,4) xdata(1:Train_Num,5) xdata(1:Train_Num),6 xdata(1:Train_Num),7]'; 
Output=xdata(1:Train_Num,2)'
%数据归一化处理
%输入数据归一化
[Inputn,In_ps]=mapminmax(Input,0,1);
%输出数据归一化处理
[Outputn,Out_ps]=mapminmax(Output,0,1);
%训练数据输入
Train_Input=Input(:,1:Train_Num);
%测试数据输入
Test_Input=Train_Input;
%训练数据输出
Train_Output=Outputn(:,1:Train_Num);
%测试数据输出
Test_Output=Train_Output;

Input_Num=5;%输入节点个数
Hidd_Num=9;%中间层隐节点胡亮取9
Out_Num=1;%网络输出维度为1
MaxEpochs=5000;%最多训练次数为5000
lr=0.01;%学习速率为0.01；
E0=0.45*10^(-2);%目标误差
W1=0.5*rand(Hidd_Num,Input_Num)-0.1;%初始化输入层与隐含层之间的权值
B1=0.5*rand(Hidd_Num,1)-0.1;%初始化输入层与隐含层之间的阈值
W2=0.5*rand(Out_Num,Hidd_Num)-0.1;%初始化输出层与隐含层之间的权值
B2=0.5*rand(Out_Num,1)-0.1;%初始化输出层与隐含层之间阈值
ErrHistory=[];%给中间变量预先占据内存

for i=1:MaxEpochs
    HiddenOut=logsig(W1*train_Input+repmat(B1,1,Train_Num));%隐含层网络输出
    NetworkOut=W*HiddenOut+repmat(B2,1,Train_Num);%输出层网络输出
    Error=Train_Output-NetworkOut;%实际输出与网络输出之差
    SSE=sumsqr(Error)%能量函数（误差平方和）
