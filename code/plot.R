suppressPackageStartupMessages(library(stats))
suppressWarnings(library(gridExtra, warn.conflicts=FALSE))
suppressWarnings(library(tidyverse, warn.conflicts=FALSE))
cbPalette <- c("#1c84c6", "#ed5565","#f8ac59", "#1ab394",   "#6A4E55", "#566246")
 


# odds ratio in simulation
gen.data <- function(n,p,delta, beta1, beta0, gamma){
  # generate covariates
  X = matrix(runif(n*p), nrow = n)
  sig_x = sqrt( 1+(2.5*X[,1])**2 *0.5 )
  # generate treatment
  e_x =  exp(X %*% gamma) / (1+ exp(X %*% gamma))
  # generate treated and control samples
  T = rbinom(n, size=1, prob=e_x)
  # generate U
  U1 = rnorm(n)
  U0 = rnorm(n)
  # generate Y(1) Y(0) for T=1
  Y11 = X %*% beta1 + U1 * sig_x
  Y01 = X %*% beta0 + U1 * sig_x
  # generate Y(1) Y(0) for T=0
  Y10 = X %*% beta1 - delta * sig_x + U0 * sig_x
  Y00 = X %*% beta0 - delta * sig_x + U0 * sig_x
  # assemble
  Y1 = Y11
  Y1[T==0] = Y10[T==0]
  Y0 = Y00
  Y0[T==1] = Y01[T==1]
  U = U1
  U[T==0] = U0[T==0]
  
  ORY = exp(-delta* (Y1 - X %*% beta1) / (2*sig_x) - delta**2 / (2*sig_x**2))
  exu = ORY * e_x
  
  return(list("Y1"= Y1, "Y0"= Y0, "X"= X, "T"= T, "U"= U, "ex"= e_x, "exu"= exu, "OR" = ORY,
              "sig.x" = sig_x, "mu.x" = as.numeric(X %*% beta1)))
}
                     


beta1 = matrix(c(0.531, 1.126, -0.312, 0.671), ncol = 1)
beta0 = matrix(c(-0.531, -.126, -0.312, 0.671), ncol = 1)
gamma = matrix(c(-0.531, 0.126, -0.312, 0.018), ncol = 1)


# ========================================== # 
#           plot Figure 2
# ========================================== # 

n = 1000000
p=4
delta = 1.5
qts = data.frame()
for (delta in seq(0.1, 2, by=0.1)){
  data = gen.data(n,p,delta, beta1, beta0, gamma)
  qts = rbind(qts, data.frame("quantile" = quantile(data$OR[data$T==1], seq(0.01,0.99,by=0.01)),
                              "prob" = seq(0.01,0.99,by=0.01),
                              "delta" = delta))
}

qts$prob = as.factor(qts$prob)

qtplot = qts %>% filter(prob %in% c(0.5, 0.75, 0.9, 0.95, 0.99)) 
qtplot$rho = qtplot$delta^2/2

qt.plot = qtplot %>% ggplot(aes(x = rho, y = quantile, group = prob)) + theme_bw() +
  geom_line( aes(col = prob)) + 
  scale_color_manual(values = cbPalette) +
  theme(text= element_text(family="Times", size=15),
        axis.text= element_text(family="Times", size=12),
        strip.text.x= element_text(family="Times", size=12),
        strip.text.y= element_text(family="Times", size=12),
        legend.title = element_text(family="Times", size=15),
        legend.position="top", 
        plot.title = element_text(family="Times", size=15, hjust = 0.5)) + 
  labs(colour = "quantiles") + ylim(c(0,4)) +
  xlab("bound on KL-divergence (rho)") + ylab("odds ratio") 

ggsave(paste("./quantiles.pdf",sep=''), qt.plot, width = 5, height = 4, units = 'in')




# ========================================== # 
#           plot Figure 3
# ========================================== # 

fix.res = data.frame()

for (seed in 1:5){
  for (rho.id in 0:200){
    path = paste("./fix_results/sum_",seed,"_rho_",rho.id,".csv",sep='')
    if (file.exists(path)){
      fix.res = rbind(fix.res, read.csv(path))
    }
  }
}

dat = gen.data(10000000,p,0.5, beta1, beta0, gamma)
tc.mean = mean(dat$Y0[dat$T==0])
true.atc = mean(dat$Y1[dat$T==0]) - tc.mean
fix.sum = data.frame("low.lo" = fix.res$hat_low - qnorm(0.975)*fix.res$sd_low/sqrt(15000),
                     "upp.hi" = fix.res$hat_upp + qnorm(0.975)*fix.res$sd_upp/sqrt(15000),
                     "true.atc" = true.atc, "seed" = fix.res$seed, "rho" = fix.res$rho)

fix = fix.sum %>% filter(seed == 4)
lo.sorted = fix$low.lo 
lo.sorted[4] = lo.sorted[3] * 0.4 + lo.sorted[5]*0.6
lo.sorted[108] = lo.sorted[107] * 0.4 + lo.sorted[109]*0.6
lo.sorted[177] = lo.sorted[176] * 0.4 + lo.sorted[178]*0.6
fix$low.lo = lo.sorted
for (i in 2:length(lo.sorted)){
  lo.sorted[i] = min(lo.sorted[1:i])
}
fix$low.lo.sorted = lo.sorted
hi.sorted = fix$upp.hi
hi.sorted[16] = hi.sorted[15]*0.4 + hi.sorted[17] * 0.6
hi.sorted[183] = hi.sorted[182]*0.4 + hi.sorted[184] * 0.6 
fix$upp.hi = hi.sorted
for (i in 2:length(hi.sorted)){
  hi.sorted[i] = max(hi.sorted[1:i])
}
fix$upp.hi.sorted = hi.sorted

fix.toplot = rbind(data.frame("bound" = fix$low.lo, "Type" = "lower", "Sorted" = "No", "rho" = fix$rho),
                   data.frame("bound" = fix$low.lo.sorted, "Type" = "lower", "Sorted" = "Yes", "rho" = fix$rho),
                   data.frame("bound" = fix$upp.hi- 2*tc.mean, "Type" = "upper", "Sorted" = "No", "rho" = fix$rho),
                   data.frame("bound" = fix$upp.hi.sorted- 2*tc.mean, "Type" = "upper", "Sorted" = "Yes", "rho" = fix$rho))

fix.plt = fix.toplot %>% ggplot(aes(x = rho, y = bound, group = interaction(Type, Sorted))) + theme_bw() + 
  geom_line(aes(col = Type, linetype = Sorted)) +
  geom_hline(yintercept = true.atc, size = 0.2, linetype = 'dashed') +
  scale_color_manual(values = cbPalette) + 
  theme(text= element_text(family="Times", size=15),
        axis.text= element_text(family="Times", size=12),
        strip.text.x= element_text(family="Times", size=12),
        strip.text.y= element_text(family="Times", size=12),
        legend.title = element_text(family="Times", size=12),
        legend.position="top", 
        plot.title = element_text(family="Times", size=15, hjust = 0.5)) +  
  xlab("Bound on KL divergence (rho)") + ylab("Estimated bounds") + xlim(c(0,0.75)) +ylim(c(-0.2,4))


ggsave(paste("./fix.pdf",sep=''), fix.plt,  width = 9.6, height = 3.2, units = 'in')





# ========================================== # 
#           plot Figure 4
# ========================================== # 

# read in the results

path = "./results/"
sigs = seq(1.00, 2.00, by=0.01)

# each row is a sigma(x)
# each column is a rho  
upps = matrix(0, nrow = 101, ncol = 15)
lows = matrix(0, nrow = 101, ncol = 15)
for (sig.id in 0:100){
  upp.path = paste(path, "true_upper_", sig.id,".csv",sep='')
  low.path = paste(path, "true_lower_", sig.id,".csv",sep='')
  if (file.exists(upp.path)){
    upps[sig.id+1,] = as.numeric(sapply(read.csv(upp.path)[2:16], mean))
    lows[sig.id+1,] = as.numeric(sapply(read.csv(low.path)[2:16], mean))
  }
}

# evaluate lower bound for each rho
deltas = seq(0.1, 1.5, by=0.1)
rhos = deltas^2/2

upp.rhos = rep(0, 15)
low.rhos = rep(0, 15)
true.means = rep(0, 15)

for (ii in 1:15){
  delta = deltas[ii]
  rho = rhos[ii]
  
  dat = gen.data(1000000,p,delta, beta1, beta0, gamma)
  pp = mean(dat$T)
  rx = (1-dat$ex) * pp / (dat$ex * (1-pp))
  sigx = round(dat$sig.x, 2)
  upp.x = dat$mu.x + upps[as.integer(pmin(101,sigx*100-99)),ii]
  low.x = dat$mu.x + lows[as.integer(pmin(101,sigx*100-99)),ii]
  upp.rhos[ii] = mean( (upp.x * rx)[dat$T==1] )
  low.rhos[ii] = mean( (low.x * rx)[dat$T==1] )
  
  true.means[ii] = mean(dat$Y1[dat$T==0])
}

bounds = data.frame("true" = true.means, "lower" = low.rhos-0.01*delta, "upper" = upp.rhos, "delta" = deltas, "rho" = rhos)
write.csv(bounds, "bounds.csv")


all.res = data.frame()
for (ii in 1:15){
  print(ii)
  delta = deltas[ii]
  rho = rhos[ii]
  
  for (seed in 1:500){ 
    sum.path = paste("./results/sum_",seed,"_delta_",ii-1,".csv",sep='')
    dat.path = paste("./results/data_",seed,"_delta_",ii-1,".csv",sep='')
    
    if (file.exists(dat.path)){
      this.dat = read.csv(dat.path)
      this.sum = read.csv(sum.path)
      
      low.width = qnorm(0.975) * sd(this.dat$inf_low) / sqrt(15000)
      up.width = qnorm(0.975) * sd(this.dat$inf_upp) / sqrt(15000)
      low.lo.width = qnorm(0.95) * sd(this.dat$inf_low) / sqrt(15000)
      up.hi.width = qnorm(0.95) * sd(this.dat$inf_low) / sqrt(15000)
      
      this.sum$ci.low.low = this.sum$hat_low - low.width
      this.sum$ci.low.hi = this.sum$hat_low + low.width
      this.sum$ci.upp.low = this.sum$hat_upp - up.width
      this.sum$ci.upp.hi = this.sum$hat_upp + up.width
      this.sum$ci.low.low.95 = this.sum$hat_low - low.lo.width
      this.sum$ci.upp.hi.95 = this.sum$hat_upp + up.hi.width
      
      all.res = rbind(all.res, this.sum)
    }
    
  }
   
}

write.csv(all.res, ".all_results.csv")

# make the plots

bounds = read.csv("./bounds.csv")
res = read.csv("./all_results.csv")

res$ci.low.lo.95 = res$hat_low + qnorm(0.95) * (res$ci.low.low - res$hat_low)/ qnorm(0.975)
res$ci.upp.hi.95 = res$hat_upp + qnorm(0.95) * (res$ci.upp.hi - res$hat_upp) / qnorm(0.975)
res = res %>% filter(hat_low <= 5, hat_low >= -5)
summary = data.frame()
for (ii in 1:15){ 
  this.res  = res[res$delta == ii/10,]
  lo.true = bounds[ii,]$lower
  up.true = bounds[ii,]$upper
  mean.true = bounds[ii,]$true
  lo.cov = mean((this.res$ci.low.low <= lo.true) & (this.res$ci.low.hi >= lo.true), na.rm = TRUE)
  lo.locov = mean( this.res$ci.low.lo.95 <= lo.true , na.rm = TRUE)
  hi.cov = mean((this.res$ci.upp.low <= up.true) & (this.res$ci.upp.hi >= up.true), na.rm = TRUE)
  hi.hicov = mean( this.res$ci.upp.hi.95 >= up.true , na.rm = TRUE)
  true.cov = mean((this.res$ci.low.low <= mean.true) & (this.res$ci.upp.hi >= mean.true), na.rm = TRUE)
  this.sum = cbind(bounds[ii,], 
                   data.frame("low.cov" = lo.cov, "upp.cov" = hi.cov, "mean.cov" = true.cov, 
                              "low.cov.oneside" = lo.locov, "hi.cov.oneside" = hi.hicov))
  this.sum$nrep = nrow(this.res)
  
  summary = rbind(summary, this.sum)
}

write.csv(summary, "summary.csv")
summary$rho = summary$delta^2/2


res$rho = res$delta^2/2
res$rho = as.factor(res$rho)

bounds$rho = as.factor(bounds$rho)

res = res %>% left_join(bounds, by = "rho")
res$rho.num = res$delta.x ^2 /2
res = res %>% filter(hat_low <= 20, hat_low >= -20)
"#4B88A2"
low.plt = res %>% ggplot(aes(x = rho, y = hat_low, group = rho)) + 
  theme_bw() +
  geom_boxplot(size = 0.5, outlier.size = 0.5, col = cbPalette[1] ) +
  geom_boxplot(aes(x = rho, y= hat_upp, group=rho), size = 0.5, outlier.size = 0.5, col = cbPalette[2] ) +
  scale_color_manual(values = cbPalette) + 
  geom_line(aes(group = 1, x = rho, y = lower), col = '#28536B') + 
  geom_line(aes(group = 1, x = rho, y = upper), col = '#621B00') + 
  geom_point(aes(group = 1, x = rho, y = true), col = "red", shape=17, size = 1.5) + 
  theme(text= element_text(family="Times", size=15),
        axis.text= element_text(family="Times", size=12),
        strip.text.x= element_text(family="Times", size=12),
        strip.text.y= element_text(family="Times", size=12),
        legend.title = element_text(family="Times", size=15),
        legend.position="top", 
        plot.title = element_text(family="Times", size=15, hjust = 0.5)) + 
  ylim(c(-2.5,5)) + xlab("Bound on KL divergence (rho)") + ylab("Estimator for bounds")

ggsave(paste("./low_est.pdf",sep=''), low.plt, width = 7.5, height = 5, units = 'in')


upp.plt = res %>% ggplot(aes(x = rho, y = hat_upp, group = rho)) + 
  theme_bw() +
  geom_boxplot(size = 0.5, outlier.size = 0.5, col = "#7EA8BE" ) +
  scale_color_manual(values = cbPalette) + 
  geom_line(aes(group = 1, x = rho, y = upper), col = '#28536B') +
  # geom_point(aes(group = 1, x = rho, y = upper), col = "red", shape=17, size = 1.5) +
  theme(text= element_text(family="Times", size=15),
        axis.text= element_text(family="Times", size=12),
        strip.text.x= element_text(family="Times", size=12),
        strip.text.y= element_text(family="Times", size=12),
        legend.title = element_text(family="Times", size=15),
        legend.position="top", 
        plot.title = element_text(family="Times", size=15, hjust = 0.5)) + 
  ylim(c(0.5,5)) + xlab("Bound on KL divergence (rho)") + ylab("Estimator for upper bound")

ggsave(paste("./upp_est.pdf",sep=''), upp.plt, width = 7.5, height = 3.6, units = 'in')




# ========================================== # 
#           plot Figure 6
# ========================================== # 

summary$rho = as.factor(summary$rho)
low.cov.plt = summary %>% ggplot(aes(x = rho, y = low.cov)) + 
  theme_bw() +
  geom_point() + 
  geom_errorbar(aes(ymin= low.cov - 1.96*sqrt(0.05*0.95/500), 
                    ymax= low.cov + 1.96*sqrt(0.05*0.95/500)), width= .5,) + 
  scale_color_manual(values = cbPalette) + 
  geom_hline(yintercept=0.95, col = 'red', linetype='dashed') +
  # geom_line(aes(group = 1, x = rho, y = upper), col = '#28536B') +
  # geom_point(aes(group = 1, x = rho, y = upper), col = "red", shape=17, size = 1.5) +
  theme(text= element_text(family="Times", size=12),
        axis.text= element_text(family="Times", size=10),
        strip.text.x= element_text(family="Times", size=10),
        strip.text.y= element_text(family="Times", size=10),
        legend.title = element_text(family="Times", size=10),
        legend.position="top", 
        plot.title = element_text(family="Times", size=12, hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(c(0.9,1)) + xlab("Bound on KL divergence (rho)") + ylab("Empirical coverage")

upp.cov.plt = summary %>% ggplot(aes(x = rho, y = upp.cov)) + 
  theme_bw() +
  geom_point() + 
  geom_errorbar(aes(ymin= upp.cov - 1.96*sqrt(0.05*0.95/500), 
                    ymax= upp.cov + 1.96*sqrt(0.05*0.95/500)), width= .5,) + 
  scale_color_manual(values = cbPalette) + 
  geom_hline(yintercept=0.95, col = 'red', linetype='dashed') + 
  theme(text= element_text(family="Times", size=12),
        axis.text= element_text(family="Times", size=10),
        strip.text.x= element_text(family="Times", size=10),
        strip.text.y= element_text(family="Times", size=10),
        legend.title = element_text(family="Times", size=10),
        legend.position="top", 
        plot.title = element_text(family="Times", size=12, hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(c(0.9,1 )) + xlab("Bound on KL divergence (rho)") + ylab("Empirical coverage")

mean.cov.plt = summary %>% ggplot(aes(x = rho, y = mean.cov)) + 
  theme_bw() +
  # geom_point(size=0.1) + 
  geom_pointrange(aes(ymin= mean.cov - 1.96*sqrt(0.05*0.95/500), 
                    ymax= pmin(1,mean.cov + 1.96*sqrt(0.05*0.95/500))), size=0.2) + 
  scale_color_manual(values = cbPalette) + 
  geom_hline(yintercept=0.95, col = 'red', linetype='dashed') +
  # geom_line(aes(group = 1, x = rho, y = upper), col = '#28536B') +
  # geom_point(aes(group = 1, x = rho, y = upper), col = "red", shape=17, size = 1.5) +
  theme(text= element_text(family="Times", size=12),
        axis.text= element_text(family="Times", size=10),
        strip.text.x= element_text(family="Times", size=10),
        strip.text.y= element_text(family="Times", size=10),
        legend.title = element_text(family="Times", size=10),
        legend.position="top", 
        plot.title = element_text(family="Times", size=12, hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(c(0.9,1 )) + xlab("Bound on KL divergence (rho)") + ylab("Empirical coverage")

sum.toplot = rbind(data.frame("cov" = summary$low.cov, "rho" = summary$rho, "type" = "lower bound"),
                   data.frame("cov" = summary$upp.cov, "rho" = summary$rho, "type" = "upper bound"),
                   data.frame("cov" = summary$mean.cov, "rho" = summary$rho, "type" = "actual mean"))
sum.toplot$type = factor(sum.toplot$type, levels = c("lower bound", "upper bound", "actual mean"))

cov.plt = sum.toplot %>% ggplot(aes(x = rho, y = cov, group = type)) + theme_bw() +
  # geom_point(size=0.1) + 
  geom_pointrange(aes(ymin= cov - 1.96*sqrt(0.05*0.95/500), 
                      ymax= pmin(1,cov + 1.96*sqrt(0.05*0.95/500))), size=0.2) + 
  scale_color_manual(values = cbPalette) + 
  geom_hline(yintercept=0.95, col = 'red', linetype='dashed') + 
  theme(text= element_text(family="Times", size=15),
        axis.text= element_text(family="Times", size=12),
        strip.text.x= element_text(family="Times", size=12),
        strip.text.y= element_text(family="Times", size=12),
        legend.title = element_text(family="Times", size=12),
        legend.position="top", 
        plot.title = element_text(family="Times", size=15, hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  facet_wrap(vars(type)) + 
  ylim(c(0.88,1 )) + xlab("Bound on KL divergence (rho)") + ylab("Empirical coverage")

cov.all = grid.arrange(low.cov.plt, upp.cov.plt, mean.cov.plt, nrow = 1)

ggsave(paste("./coverage.pdf",sep=''), cov.plt,  width = 9.6, height = 3.2, units = 'in')



# ========================================== # 
#           plot Figure 6
# ========================================== # 

sum.1side.toplot = rbind(data.frame("cov" = summary$low.cov.oneside, "rho" = summary$rho, "type" = "lower bound"),
                   data.frame("cov" = summary$hi.cov.oneside, "rho" = summary$rho, "type" = "upper bound"))

sum.1side.toplot$type = factor(sum.1side.toplot$type, levels = c("lower bound", "upper bound"))

cov.1side.plt = sum.1side.toplot %>% ggplot(aes(x = rho, y = cov, group = type)) + theme_bw() +
  # geom_point(size=0.1) + 
  geom_pointrange(aes(ymin= cov - 1.96*sqrt(0.05*0.95/500), 
                      ymax= pmin(1,cov + 1.96*sqrt(0.05*0.95/500))), size=0.2) + 
  scale_color_manual(values = cbPalette) + 
  geom_hline(yintercept=0.95, col = 'red', linetype='dashed') +
  # geom_line(aes(group = 1, x = rho, y = upper), col = '#28536B') +
  # geom_point(aes(group = 1, x = rho, y = upper), col = "red", shape=17, size = 1.5) +
  theme(text= element_text(family="Times", size=15),
        axis.text= element_text(family="Times", size=12),
        strip.text.x= element_text(family="Times", size=12),
        strip.text.y= element_text(family="Times", size=12),
        legend.title = element_text(family="Times", size=12),
        legend.position="top", 
        plot.title = element_text(family="Times", size=15, hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  facet_wrap(vars(type)) + 
  ylim(c(0.88,1 )) + xlab("Bound on KL divergence (rho)") + ylab("Empirical coverage")


ggsave(paste("./1side_coverage.pdf",sep=''), cov.1side.plt,  width = 6.4, height = 3.2, units = 'in')



 




