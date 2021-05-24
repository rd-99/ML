using CSV
using DataFrames
dataset=CSV.read("housingPriceData.csv")
price=dataset.price
bathno=dataset.bathrooms
bedno=dataset.bedrooms
sqrt_ft=dataset.sqft_living
function mean(B)
    avg=0
    for i in 1:length(B)
        avg=avg+B[i]
    end
    avg=avg/length(B)
    return avg
end
function std_dev(B)
    avg=mean(B)
    sum=0
    for i in 1:length(B)
        sum+=(B[i]^2)
    end
    sd=((sum/length(B))-(avg^2))
    return sqrt(sd)
end
function data_process(B)
    avg=mean(B)
    sd=std_dev(B)
    C=zeros(length(B))
    for i in 1:length(B)
        C[i]=(B[i]-avg)/sd
    end
    return C
end
bedpro=data_process(bedno)
bathpro=data_process(bathno)
sqft_proc=data_process(sqrt_ft)
k=length(price)
k_tr=convert(Int64,round(0.6*k,digits=0))
k_vd=convert(Int64,round(0.8*k,digits=0))
m_te=k-k_tr-k_vd
x0=ones(k_tr)
x_vd=ones(k_vd-k_tr+1)
X=cat(x0,bedpro[1:k_tr],bathpro[1:k_tr],sqft_proc[1:k_tr],dims=2)
Y=price[1:k_tr]
B=zeros(4)
function costfn(X,Y,B,lam)
    m=length(Y)
    cost=(sum((Y-(X*B)).^2)/(2*m))+(lam*(sum(B.^2))-lam*(B[1]^2))
    return cost
end
function grad(X,Y,B,al,lam,iter)
    costhistory=zeros(iter)
    m=length(price)
    for i in 1:iter
        mod1=(X'*(Y-(X*B)))./m
        mod2=B.*((-2)*lam)
        mod=(mod1+mod2)
        B[2:end]=B[2:end]+(mod[2:end]*al)
        B[1]=B[1]+mod1[1]*al
        costhistory[i]=costfn(X,Y,B,lam)
    end
    return B,costhistory
end
al=0.0001
X_pred=cat(x_vd,bedpro[k_tr:k_vd],bathpro[k_tr:k_vd],sqft_proc[k_tr:k_vd],dims=2)
newB,costHistory=grad(X,Y,B,al,0,10000)
Y_pred=X_pred*newB
min_cost=costfn(X_pred,price[k_tr:k_vd],newB,0)
opt_lambda=0
print(min_cost)
for i in 0:0.01:0.5
    newB,costHistory=grad(X,Y,B,al,i,10000)
    Y_pred=X_pred*newB
    net_cost=costfn(X_pred,price[k_tr:k_vd],newB,i)
    if net_cost<min_cost
        min_cost=net_cost
        opt_lambda=i
    end
end
print(opt_lambda)
newB,costHistory=grad(X,Y,B,al,0.02,10000)
X_pred=cat(ones(k-k_vd+1),bedpro[k_vd:end],bathpro[k_vd:end],sqft_proc[k_vd:end],dims=2)
Y_pred=X_pred*newB
print(newB)
print("\nCoefficients are:\n")
print("B0 = ")
print(newB[1])
print("\n")
print("B1 = ")
print(newB[2])
print("\n")
print("B2 = ")
print(newB[3])
print("\n")
print("B3 = ")
print(newB[4]/1000)
print("\n")
function rms_error(Y,Y_pred)
    m=length(Y)
    rms=sum((Y-Y_pred).^2)/m
    return (sqrt(rms))
end
print("RMS error = ")
print(rms_error(price[k_vd:end],Y_pred))
function r2_score(Y,Y_pred)
    m=length(Y)
    mean=sum(Y_pred)/m
    mean_Y=ones(m).*mean
    score=sum((Y_pred-Y).^2)/sum((Y-mean_Y).^2)
    score=1-score
end
print("R2 Score = ")
print(r2_score(price[k_vd:end],Y_pred))
print(rms_error(price[k_vd:end],Y_pred))
print("\n")
print(r2_score(price[k_vd:end],Y_pred))
print("\n")
print(newB)
df=DataFrame(Predicted_Price=Y_pred[:])
CSV.write("2a.csv",df)
