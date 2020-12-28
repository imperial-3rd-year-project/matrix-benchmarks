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
  , evalNet
  ) where


import Data.Array.Accelerate              as A hiding ((!!), length)
import Data.Array.Accelerate.Numeric.LinearAlgebra hiding (trace)
import Prelude as P
import Data.Array.Accelerate.LLVM.Native as CPU

-- Utilities

evalNet net = undefined -- let BasicNetwork ws bs = net
                -- in BasicNetwork (P.map (use . CPU.run) ws) (P.map (use . CPU.run) bs)

(^+^) :: (Shape sh, P.Num (Exp c), Elt c) => Acc (Array sh c) -> Acc (Array sh c) -> Acc (Array sh c)
u ^+^ v = A.zipWith (+) u v 

(^-^) :: (Shape sh, P.Num (Exp c), Elt c) => Acc (Array sh c) -> Acc (Array sh c) -> Acc (Array sh c)
u ^-^ v = A.zipWith (-) u v 

(^*^) :: (Shape sh, P.Num (Exp c), Elt c) => Acc (Array sh c) -> Acc (Array sh c) -> Acc (Array sh c)
u ^*^ v = A.zipWith (*) u v

(^+^^) :: (Shape sh, P.Num (Exp c), Elt c) => (Array sh c) -> (Array sh c) -> (Array sh c)
(^+^^) = runN (^+^)

(*^) :: forall sh a. (Shape sh, Elt a, P.Num (Exp a)) => Exp a -> Acc (Array sh a) -> Acc (Array sh a) 
s *^ v = A.map (\x -> x * s) v

type Activation = Exp Double -> Exp Double

sigmoid :: Activation
sigmoid = \z -> 1.0 / (1.0 + exp (-z))

sigmoid' :: Exp Double -> Exp Double
sigmoid' = \z -> sigmoid z * (1 - sigmoid z)

data BasicNetwork = BasicNetwork [Matrix Double] [Vector Double]
  deriving Show


create :: [Int] -> BasicNetwork
create xs = BasicNetwork weights biases
  where
    weights :: [Matrix Double]
    weights = do
      idx <- [1..(length xs - 1)]
      pure $ (fromList ( Z :. xs!!idx :. xs!!(idx - 1) ) [1..] :: Matrix Double)
    
    biases :: [Vector Double]
    biases = do
      idx <- [1..(length xs - 1)]
      pure $ (fromList (Z :. xs!!idx) [1..] :: Vector Double)


feedforward :: BasicNetwork -> Acc (Vector Double) -> Acc (Vector Double)
feedforward (BasicNetwork ws bs) input = res
  where
    ws' = P.map use ws
    bs' = P.map use bs
    res = feedforward' ws' bs' input
    feedforward' :: [ Acc (Matrix Double) ] -> [ Acc (Vector Double) ] -> Acc (Vector Double) -> Acc (Vector Double)
    feedforward' [] [] a = a
    feedforward' (w:ws) (b:bs) a = feedforward' ws bs $ A.map sigmoid $ (w #> a) ^+^ b

type TrainingData = [(Vector Double, Vector Double)]

train :: BasicNetwork -> Int -> TrainingData -> Int -> Double -> BasicNetwork
train net n td epochs eta = loop epochs net
  where
    loop 0 net = net
    loop !epochs net@(BasicNetwork weights biases) = loop (epochs - 1) net'
      where
        net' = BasicNetwork weights' biases'
        
        nablaB :: [ Vector Double ]
        nablaW :: [ Matrix Double ]
        (nablaB, nablaW) = descend td
        
        --         training data     biases                 weights               for each layer
        descend :: TrainingData -> ([Vector Double], [Matrix Double])
        descend [(x, y)]     = backprop x y net
        descend ((x, y):td') = (nablaB', nablaW')
          where
            (nablaB, nablaW) = descend td'
            (deltaNablaB, deltaNablaW) = backprop x y net
            nablaB' = P.map (P.uncurry (^+^^)) $ P.zip nablaB deltaNablaB
            nablaW' = P.map (P.uncurry (^+^^)) $ P.zip nablaW deltaNablaW
        
        weights' = P.map (P.uncurry goodLongass) $ P.zip weights nablaW
        biases'  = P.map (P.uncurry goodLongass) $ P.zip biases nablaB

    velocity :: Exp Double
    velocity = lift (eta / P.fromIntegral n)
    goodLongass :: forall sh. (Shape sh) => Array sh Double -> Array sh Double -> Array sh Double
    goodLongass = runN (longass velocity)

longass :: forall sh a. (Shape sh, Elt a, P.Num (Exp a)) => Exp a -> Acc (Array sh a) -> Acc (Array sh a) -> Acc (Array sh a) 
longass velocity w wb = w ^-^ (velocity *^ wb)

backpropLayer :: Acc (Matrix Double) -- weight
              -> Acc (Vector Double) -- activation
              -> Acc (Vector Double) -- z
              -> Acc (Vector Double) -- delta
              -> Acc (Vector Double, Matrix Double) -- delta, new weight
backpropLayer w a z delta = lift (delta', y)
  where
    sp             = A.map sigmoid' z
    delta'         = ((transpose w) #> delta) ^*^ sp
    y              = delta' >< a

backpropLayer' = runN backpropLayer

backpropLastLayer :: Acc (Vector Double)
                  -> Acc (Vector Double)
                  -> Acc (Vector Double)
                  -> Acc (Vector Double)
                  -> Acc (Vector Double, Matrix Double)
backpropLastLayer expected a' a z = lift (delta, nw)
  where
    delta = (cost' a expected) ^*^ (A.map sigmoid' z)
    nw = delta >< a'

backpropLastLayer' = runN backpropLastLayer

backprop :: Vector Double -> Vector Double -> BasicNetwork -> ([Vector Double], [Matrix Double])
backprop actual expected (BasicNetwork ws bs) = (b, w)
  where
    (b, w) = backprop' (P.tail ws) activations zs
    backprop' :: [Matrix Double] 
              -> [Vector Double] 
              -> [Vector Double] 
              -> ([Vector Double], [Matrix Double])
    backprop' [] [a', a] [z] = ([delta], [nw])
      where
        (delta, nw) = backpropLastLayer' expected a' a z
    backprop' (w:ws) (a:as) (z:zs) = (delta':delta:xs, y:ys)
      where
        (delta', y)    = backpropLayer' w a z delta
        (delta:xs, ys) = backprop' ws as zs

    (activations, zs) = calcActivations actual ws bs
    
    calcActivations x' [] [] = ([x'], [])
    calcActivations x' (w:ws) (b:bs) = (x':as, z:zs)
      where
        (x'', z) = calcActivationLayer' x' w b
        (as, zs) = calcActivations x'' ws bs

calcActivationLayer :: Acc (Vector Double) -> Acc (Matrix Double) -> Acc (Vector Double) -> Acc (Vector Double, Vector Double)
calcActivationLayer x weight bias = let z = (weight #> x) ^+^ bias in lift (A.map sigmoid z, z)

calcActivationLayer' = runN calcActivationLayer

cost' :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
cost' actual expected = actual ^-^ expected
