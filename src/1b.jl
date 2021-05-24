using CSV
using DataFrames
dataset=CSV.read("housingPriceData.csv")
price=dataset.price
bathno=dataset.bathrooms
bedno=dataset.bedrooms
function mean(P)
    avg=0
    for i in 1:length(P)
        avg=avg+P[i]
    end
    avg=avg/length(P)
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
k=length(price)
k_training=convert(Int64,round(0.8*k,digits=0))
k_test=length(price)-k_training
initial=ones(k_training)
Y=price[1:k_training]
X=cat(initial,data_process(bedno[1:k_training]),data_process(bathno[1:k_training]),data_process((bedno[1:k_training]).^2),data_process((bathno[1:k_training]).^2),data_process((bedno[1:k_training]).*(bathno[1:k_training])),dims=2)
B=zeros(6,1)
function costfn(X,Y,B)
    m=length(Y)
    cost=sum((Y-(X*B)).^2)/(2*m)
    return cost
end
function grad_descent(X,Y,B,al,iter)
    costhistory=zeros(iter)
    m=length(Y)
    for i in 1:iter
        mod=(X'*(Y-(X*B)))
        B=B+(mod*al)./m
        costhistory[i]=costfn(X,Y,B)
    end
    return B,costhistory
end
al=0.0001
newB,costHistory=grad_descent(X,Y,B,al,10000)
initial=ones(k-k_training+1)
X_test=cat(initial,data_process(bedno[k_training:k]),data_process(bathno[k_training:k]),data_process((bedno[k_training:k]).^2),data_process((bathno[k_training:k]).^2),data_process((bedno[k_training:k]).*(bathno[k_training:k])),dims=2)
Y_prediction=X_test*newB
function r2_score(Y,Y_prediction)
    m=length(Y)
    avg=sum(Y_prediction)/m
    avg_Y=ones(m).*avg
    score=sum((Y_prediction-Y).^2)/sum((Y-avg_Y).^2)
    score=1-score
end
function rms_error(Y,Y_prediction)
    m=length(Y)
    rms=sum((Y-Y_prediction).^2)/m
    return (sqrt(rms))
end
print(rms_error(price[k_training:k],Y_prediction))
print("\n")
print(r2_score(price[k_training:k],Y_prediction))
print("\n")
print(newB)
df=DataFrame(Predicted_Price=Y_prediction[:])
CSV.write("1b.csv",df)
