install.packages("Amelia")
library(Amelia)
#function to read amelia file and return imputed dataframe
ameliaImputer <- function(df){
  a.out <- amelia(x = "df",m=1)
  write.amelia(a.out ,file.stem = "imputed_data",format="csv")
  imputed_data=read.csv("imputed_data.csv")
  return(imputed_data)
}
#calling the function on the dataset
df_imputed <-ameliaImputer(df)