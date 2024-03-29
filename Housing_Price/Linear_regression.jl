using CSV
using DataFrames
dataset=CSV.read("housingPriceData.csv")
price=dataset.price
bedno=dataset.bedrooms
bathno=dataset.bathrooms
sqrt_ft=dataset.sqft_living
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
bedpro=data_process(bedno)
bathpro=data_process(bathno)
sqft_proc=data_process(sqrt_ft)
k=length(price)
k_training=convert(Int64,round(0.8*k,digits=0))
k_test=length(price)-k_training
initial=ones(k_training)
X=cat(initial,bedpro[1:k_training],bathpro[1:k_training],sqft_proc[1:k_training],dims=2)
Y=price[1:k_training]
B=zeros(4)
function cost_fn(X,Y,B)
    m=length(Y)
    cost=sum(((X*B)-Y).^2)/(2*m)
    return cost
end
function grad_descent(X,Y,B,al,iter)
    costHistory=zeros(iter)
    m=length(Y)
    for iteration in 1:iter
        H=X*B
        loss=H-Y
        grad=(X'*loss)/m
        B=B-al*grad
        cost=cost_fn(X,Y,B)
        costHistory[iteration]=cost
    end
    return B,costHistory
end
al=0.0001
newB,costHistory=grad_descent(X,Y,B,al,10000)
initial=ones(k-k_training+1)
X_test=cat(initial,bedpro[k_training:k],bathpro[k_training:k],sqft_proc[k_training:k],dims=2)
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
df=DataFrame(Predicted_Price=Y_prediction)
CSV.write("1a.csv",df)
