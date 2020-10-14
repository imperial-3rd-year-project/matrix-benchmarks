{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE BangPatterns #-}


module Network2 (
    BasicNetwork
  , feedforward
  , create
  , train
  ) where


import Numeric.LinearAlgebra as NLA
import Debug.Trace

-- Utilities

type Activation = Double -> Double

sigmoid :: Activation
sigmoid = \z -> 1.0 / (1.0 + exp (-z))

sigmoid' :: R -> R
sigmoid' = \z -> sigmoid z * (1 - sigmoid z)

data BasicNetwork = BasicNetwork [Matrix R] [Vector R]
  deriving Show

create :: [Int] -> BasicNetwork
create xs = BasicNetwork weights biases
  where
    weights :: [Matrix R]
    weights = [(((xs!!idx) >< (xs!!(idx-1)))) [1..] | idx <- [1..(length xs - 1)]]
    
    biases :: [Vector R]
    biases = [ fromList $ take (xs!!idx) [1..] | idx <- [1..(length xs - 1)] ]


feedforward :: BasicNetwork -> (Vector R) -> (Vector R)
feedforward (BasicNetwork ws bs) input = res
  where
    res = feedforward' ws bs input
    feedforward' :: [ Matrix R ] -> [Vector R] -> (Vector R) -> (Vector R)
    feedforward' [] [] a = a
    feedforward' (w:ws) (b:bs) a = feedforward' ws bs $ cmap sigmoid $ (w #> a) + b

type TrainingData = [ (Vector R, Vector R) ]

train :: BasicNetwork -> Int -> TrainingData -> Int -> Double -> BasicNetwork
train net _ _ 0 _         = net
train net n td epochs eta = train net' n td (epochs - 1) eta
  where
    BasicNetwork weights biases = net
    net' = BasicNetwork weights' biases'
    
    nablaB :: [ Vector R ]
    nablaW :: [ Matrix R ]
    (nablaB, nablaW) = descend td
    
    --         training data     biases      weights   for each layer
    descend :: TrainingData -> ([Vector R], [Matrix R])
    descend [(x, y)]     = backprop x y net
    descend ((x, y):td') = (nablaB', nablaW')
      where
        (nablaB, nablaW) = descend td'
        (deltaNablaB, deltaNablaW) = backprop x y net
        nablaB' = [ nb + dnb | (nb, dnb) <- zip nablaB deltaNablaB ]
        nablaW' = [ nw + dnw | (nw, dnw) <- zip nablaW deltaNablaW ]
    
    velocity = eta / fromIntegral n
    weights' = [w - (cmap ((*) velocity) wb) | (w, wb) <- zip weights nablaW]
    biases'  = [b - (cmap ((*) velocity) nb) | (b, nb) <- zip biases  nablaB]


backprop :: Vector R -> Vector R -> BasicNetwork -> ([Vector R], [Matrix R])
backprop actual expected (BasicNetwork ws bs) = (b, w)
  where
    (b, w) = backprop' (tail ws) activations zs
    backprop' :: [Matrix R] 
              -> [Vector R] 
              -> [Vector R] 
              -> ([Vector R], [Matrix R])
    backprop' [] [a', a] [z] = ([delta], [nw])
      where
        delta = (cost' a expected) * (cmap sigmoid' z)
        nw = delta `outer` a'
    backprop' (w:ws) (a:a':as) (z:zs) = (delta':delta:xs, y:ys)
      where
        sp             = cmap sigmoid' z
        delta'         = ((tr w) #> delta) * sp
        y              = delta' `outer` a
        (delta:xs, ys) = backprop' ws (a':as) zs

    (activations, zs) = calcActivations actual ws bs
    
    calcActivations :: Vector R -> [Matrix R] -> [Vector R] -> ([Vector R], [Vector R])
    calcActivations x' [] [] = ([x'], [])
    calcActivations x' (w:ws) (b:bs) = (x':as, z:zs)
      where
        (as, zs) = calcActivations x'' ws bs
        z        = (w #> x') + b :: Vector R
        x''      = cmap sigmoid z

cost' :: Vector R -> Vector R -> Vector R
cost' actual expected = actual - expected
