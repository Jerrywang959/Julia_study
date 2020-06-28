# char-rnn.jl
## using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 1e-2	# Learning rate
    seqlen::Int = 50	# Length of batchseqences
    nbatch::Int = 50	# number of batches text is divided into
    throttle::Int = 30	# Throttle timeout
end




args=Args()

isfile("input.txt") ||
download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt","input.txt")

text = collect(String(read("input.txt")))

# an array of all unique characters
alphabet = [unique(text)..., '_']

text = map(ch -> Flux.onehot(ch, alphabet), text)
stop = Flux.onehot('_', alphabet)

N = length(alphabet)

# Partitioning the data as sequence of batches, which are then collected as array of batches
Xs = collect(partition(Flux.batchseq(Flux.chunk(text, args.nbatch), stop), args.seqlen))
Ys = collect(partition(Flux.batchseq(Flux.chunk(text[2:end], args.nbatch), stop), args.seqlen))
