install.packages("quantmod")
library("quantmod")

### specify ticker names and download the data
tickers = c("^OMX", "ABB.ST", "NDA-SE.ST")
getSymbols(tickers, src="yahoo", from = '2015-01-01', to = "2021-11-01")

###### get adjusted prices and save them in market_data
OMX_ad = OMX$OMX.Adjusted
ABB.ST_ad = ABB.ST$ABB.ST.Adjusted
###### a  modification is needed in the case of Nordea because of '-' in the name
NDA_SE.ST = `NDA-SE.ST`
NDA_SE.ST_ad = NDA_SE.ST$`NDA-SE.ST.Adjusted`
market_data=cbind(OMX_ad,ABB.ST_ad,NDA_SE.ST_ad)

##### remove NA
market_data=na.omit(market_data)

##### compute log-returns and save them as data frame called "returns", whose colums are renamed, respectively.
k=nrow(market_data)
returns=as.data.frame(log(as.matrix(market_data[2:k,]))-log(as.matrix(market_data[1:(k-1),])))
names(returns)=c("rOMX","rABB","rNDA.SE.ST")