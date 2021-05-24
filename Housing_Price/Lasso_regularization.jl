using CSV
using DataFrames
dataset=CSV.read("housingPriceData.csv")
price=dataset.price
bathno=dataset.bathrooms
bedno=dataset.bedrooms
sqrt_ft=dataset.sqft_living
function mean(B)
    mean=0
    for i in 1:length(B)
        mean+=B[i]
    end
    mean/=length(B)
    return mean
end
function std_dev(B)
    temp=mean(B)
    sum=0
    for i in 1:length(B)
        sum+=(B[i]^2)
    end
    dev=((sum/length(B))-(temp^2))
    return sqrt(dev)
end
function data_process(B)
    temp1=mean(B)
    temp2=std_dev(B)
    C=zeros(length(B))
    for i in 1:length(B)
        C[i]=(B[i]-temp1)/temp2
    end
    return C
end
bedpro=data_process(bedno)
bathpro=data_process(bathno)
sqft_proc=data_process(sqrt_ft)
m=length(price)
x0=ones(m)
X=cat(x0,bedpro,bathpro,sqft_proc,dims=2)
Y=price
B=zeros(4)
function abs_sum(B)
    sum=0
    for i in 1:length(B)
        if B[i]>0
            sum=sum+B[i]
        else
            sum=sum-B[i]
        end
    end
    return sum
end
function costfn(X,Y,B,lam)
    m=length(Y)
    cost=(sum((Y-(X*B)).^2)/(2*m)+lam*(abs_sum(B[2:end])))
    return cost
end
lam=0.1
function abs_deriv(B)
    l=zeros(length(B))
    for i in 1:length(B)
        if B[i]>0
            l[i]=1
        elseif B[i]<0
            l[i]=-1
        else
            l[i]=0
        end
    end
    return l
end
function grad(X,Y,B,al,lam,iterations)
    costhistory=zeros(iterations)
    m=length(price)
    for i in 1:iterations
        mod1=(X'*(Y-(X*B)))./m
        l=abs_deriv(B)
        mod2=l.*((-1)*lam)
        mod=(mod1+mod2)
        B[2:end]=B[2:end]+(mod[2:end]*al)
        B[1]=B[1]+(mod1[1]*al)
        costhistory[i]=costfn(X,Y,B,lam)
    end
    return B,costhistory
end
al=0.0001
lam=0.5
newB,costHistory=grad(X,Y,B,al,lam,100000)
function rms_error(Y,Y_pred)
    m=length(Y)
    rms=sum((Y-Y_pred).^2)/m
    return (sqrt(rms))
end
function r2_score(Y,Y_pred)
    m=length(Y)
    mean=sum(Y_pred)/m
    mean_Y=ones(m).*mean
    score=sum((Y_pred-Y).^2)/sum((Y-mean_Y).^2)
    score=1-score
end
Y_pred=X*newB
print(rms_error(Y,Y_pred))
print("\n")
print(r2_score(Y,Y_pred))
print("\n")
print(newB)
df=DataFrame(Predicted_Price=Y_pred[:])
CSV.write("2b.csv",df)
