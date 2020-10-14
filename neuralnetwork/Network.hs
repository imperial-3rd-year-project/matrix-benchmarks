{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE BangPatterns #-}


module Network (
    BasicNetwork
  , feedforward
  , create
  , train
  ) where


import Data.Array.Accelerate              as A hiding ((!!), length)
import Data.Array.Accelerate.Numeric.LinearAlgebra hiding (trace)
import Prelude as P

import Data.Array.Accelerate.LLVM.Native  as CPU

import Debug.Trace

-- Utilities


(^+^) :: (Shape sh, P.Num (Exp c), Elt c) => Acc (Array sh c) -> Acc (Array sh c) -> Acc (Array sh c)
u ^+^ v = A.zipWith (+) u v 

(^-^) :: (Shape sh, P.Num (Exp c), Elt c) => Acc (Array sh c) -> Acc (Array sh c) -> Acc (Array sh c)
u ^-^ v = A.zipWith (-) u v 

(^*^) :: (Shape sh, P.Num (Exp c), Elt c) => Acc (Array sh c) -> Acc (Array sh c) -> Acc (Array sh c)
u ^*^ v = A.zipWith (*) u v

(*^) :: forall sh a. (Shape sh, Elt a, P.Num (Exp a)) => Exp a -> Acc (Array sh a) -> Acc (Array sh a) 
s *^ v = A.map (\x -> x * s) v

type Activation = Exp Double -> Exp Double

sigmoid :: Activation
sigmoid = \z -> 1.0 / (1.0 + exp (-z))

sigmoid' :: Exp Double -> Exp Double
sigmoid' = \z -> sigmoid z * (1 - sigmoid z)

data BasicNetwork = BasicNetwork [Acc (Matrix Double)] [Acc (Vector Double)]
  deriving Show

-- Force evaluation of the array and network

evalNet :: BasicNetwork -> BasicNetwork
evalNet (BasicNetwork ws bs) = BasicNetwork (P.map compute ws) (P.map compute bs)


create :: [Int] -> BasicNetwork
create xs = BasicNetwork weights biases
  where
    weights :: [Acc (Matrix Double)]
    weights = do
      idx <- [1..(length xs - 1)]
      pure $ use $ (fromList ( Z :. xs!!idx :. xs!!(idx - 1) ) [1..] :: Matrix Double)
    
    biases :: [Acc (Vector Double)]
    biases = do
      idx <- [1..(length xs - 1)]
      pure $ use $ (fromList (Z :. xs!!idx) [1..] :: Vector Double)


feedforward :: BasicNetwork -> Acc (Vector Double) -> Acc (Vector Double)
feedforward (BasicNetwork ws bs) input = res
  where
    res = feedforward' ws bs input
    feedforward' :: [ Acc (Matrix Double) ] -> [ Acc (Vector Double) ] -> Acc (Vector Double) -> Acc (Vector Double)
    feedforward' [] [] a = a
    feedforward' (w:ws) (b:bs) a = feedforward' ws bs $ A.map sigmoid $ (w #> a) ^+^ b

type TrainingData = [ (Acc (Vector Double), Acc (Vector Double)) ]

train :: BasicNetwork -> Int -> TrainingData -> Int -> Double -> BasicNetwork
train net _ _ 0 _         = net
train net n td epochs eta = train net' n td (epochs - 1) eta
  where
    BasicNetwork weights biases = net
    net' = evalNet $ BasicNetwork weights' biases'
    
    nablaB :: [ Acc (Vector Double) ]
    nablaW :: [ Acc (Matrix Double) ]
    (nablaB, nablaW) = descend td
    
    --         training data     biases                 weights               for each layer
    descend :: TrainingData -> ([Acc (Vector Double)], [Acc (Matrix Double)])
    descend [(x, y)]     = backprop x y net
    descend ((x, y):td') = (nablaB', nablaW')
      where
        (nablaB, nablaW) = descend td'
        (deltaNablaB, deltaNablaW) = backprop x y net
        nablaB' = [ nb ^+^ dnb | (nb, dnb) <- P.zip nablaB deltaNablaB ]
        nablaW' = [ nw ^+^ dnw | (nw, dnw) <- P.zip nablaW deltaNablaW ]
    
    velocity = lift (eta / P.fromIntegral n)
    weights' = [w ^-^ (velocity *^ wb) | (w, wb) <- P.zip weights nablaW]
    biases'  = [b ^-^ (velocity *^ nb) | (b, nb) <- P.zip biases  nablaB]


backprop :: Acc (Vector Double) -> Acc (Vector Double) -> BasicNetwork -> ([Acc (Vector Double)], [Acc (Matrix Double)])
backprop actual expected (BasicNetwork ws bs) = (P.map compute b, P.map compute w)
  where
    (b, w) = backprop' (P.tail ws) activations zs
    backprop' :: [Acc (Matrix Double)] 
              -> [Acc (Vector Double)] 
              -> [Acc (Vector Double)] 
              -> ([Acc (Vector Double)], [Acc (Matrix Double)])
    backprop' [] [a', a] [z] = ([delta], [nw])
      where
        delta = (cost' a expected) ^*^ (A.map sigmoid' z)
        nw = delta >< a'
    backprop' (w:ws) (a:a':as) (z:zs) = (delta':delta:xs, y:ys)
      where
        sp             = A.map sigmoid' z
        delta'         = ((transpose w) #> delta) ^*^ sp
        y              = delta' >< a
        (delta:xs, ys) = backprop' ws (a':as) zs

    (activations, zs) = calcActivations actual ws bs
    
    calcActivations x' [] [] = ([x'], [])
    calcActivations x' (w:ws) (b:bs) = (x':as, z:zs)
      where
        (as, zs) = calcActivations x'' ws bs
        z        = (w #> x') ^+^ b
        x''      = A.map sigmoid z

cost' :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
cost' actual expected = actual ^-^ expected
