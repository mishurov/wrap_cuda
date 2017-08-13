library(ggplot2)

benchmarks = read.csv("benchmarks.csv")

vertices <- unique(benchmarks$Vertices)
len <- length(vertices)

transfVals <- benchmarks$Transfer
transfVals <- transfVals[transfVals != 0]

compVals <- benchmarks$Computation
compVals <- compVals[compVals != 0]

gpuVals <- benchmarks[benchmarks$Type == "GPU",]$Time
cpuVals <- benchmarks[benchmarks$Type == "CPU",]$Time

transfLabels <- rep("GPU memory transfers", len)
compLabels <- rep("GPU computations", len)
gpuLabels <- rep("GPU total", len)
cpuLabels <- rep("CPU total", len)

transferFrame <- data.frame(Type=transfLabels,
                           Vertices=vertices,
                           Time=transfVals)
computationFrame <- data.frame(Type=compLabels,
                              Vertices=vertices,
                              Time=compVals)
gpuFrame <- data.frame(Type=gpuLabels,
                       Vertices=vertices,
                       Time=gpuVals)
cpuFrame <- data.frame(Type=cpuLabels,
                       Vertices=vertices,
                       Time=cpuVals)


tt <- rbind(cpuFrame, gpuFrame)
p <- ggplot(tt, aes(x = Vertices, y = Time))
p <- p + geom_line(aes(color = Type))
print(p)


cg <- rbind(cpuFrame, computationFrame, transferFrame)
p <- ggplot(cg, aes(x = Vertices, y = Time))
p <- p + geom_line(aes(color = Type))
print(p)

