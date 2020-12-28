{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

module UnrolledNetwork where

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

(*^) :: forall sh a. (Shape sh, Elt a, P.Num (Exp a)) => Exp a -> Acc (Array sh a) -> Acc (Array sh a) 
s *^ v = A.map (\x -> x * s) v

type Activation = Exp Double -> Exp Double

sigmoid :: Activation
sigmoid = \z -> 1.0 / (1.0 + exp (-z))

sigmoid' :: Exp Double -> Exp Double
sigmoid' = \z -> sigmoid z * (1 - sigmoid z)

type Layer = (Matrix Double, Vector Double)
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
    res = feedforward' (P.map use ws) (P.map use bs) input
    feedforward' :: [ Acc (Matrix Double) ] -> [ Acc (Vector Double) ] -> Acc (Vector Double) -> Acc (Vector Double)
    feedforward' [] [] a = a
    feedforward' (w:ws) (b:bs) a = feedforward' ws bs $ A.map sigmoid $ (w #> a) ^+^ b

type TrainingData = [ (Acc (Vector Double), Acc (Vector Double)) ]

train' :: Int -> Double
       -> TrainingData
       -> Acc Layer
       -> Acc Layer
       -> Acc Layer
       -> Acc (Layer, Layer, Layer)
train' n eta td layer1 layer2 layer3 = lift (newLayer layer1 n1, newLayer layer2 n2, newLayer layer3 n3)
  where
    descend :: TrainingData -> Acc (Layer, Layer, Layer)
    descend [(x, y)]     = backprop x y layer1 layer2 layer3
    descend ((x, y):td') = lift (toNabla n1 d1, toNabla n2 d2, toNabla n3 d3)
      where
        (n1, n2, n3) :: (Acc Layer, Acc Layer, Acc Layer) = unlift (descend td')
        (d1, d2, d3) :: (Acc Layer, Acc Layer, Acc Layer) = unlift (backprop x y layer1 layer2 layer3)
        toNabla :: Acc Layer -> Acc Layer -> Acc Layer
        toNabla n d = lift (wn ^+^ wd, bn ^+^ bd)
          where
            (wn, bn) = unlift n
            (wd, bd) = unlift d

    (n1, n2, n3) :: (Acc Layer, Acc Layer, Acc Layer) = unlift (descend td)

    velocity :: Exp Double
    velocity = lift (eta / P.fromIntegral n)
    newLayer :: Acc Layer -> Acc Layer -> Acc Layer
    newLayer layer nabla = lift (lw ^-^ (velocity *^ nw), lb ^-^ (velocity *^ nb))
      where
        (lw, lb) = unlift layer
        (nw, nb) = unlift nabla


train :: BasicNetwork -> Int -> TrainingData -> Int -> Double -> BasicNetwork
train (BasicNetwork [w1, w2, w3] [b1, b2, b3]) n td epochs eta =
  let ((w1', b1'), (w2', b2'), (w3', b3')) = loop ((w1, b1), (w2, b2), (w3, b3)) epochs
   in BasicNetwork [w1', w2', w3'] [b1', b2', b3']
  where
    loop net 0 = net
    loop (l1, l2, l3) epochs =
      loop (train'' l1 l2 l3) (epochs - 1)

    train'' = runN (train' n eta td)

backprop :: Acc (Vector Double)
         -> Acc (Vector Double)
         -> Acc Layer -> Acc Layer -> Acc Layer
         -> Acc (Layer, Layer, Layer)
backprop actual expected layer1 layer2 layer3 = lift (lift (w1', b1') :: Acc Layer
                                                    , lift (w2', b2') :: Acc Layer,
                                                      lift (w3', b3') :: Acc Layer)
  where
    (w1, b1) = unlift layer1
    (w2, b2) = unlift layer2
    (w3, b3) = unlift layer3
    ws = [w1, w2, w3]
    bs = [b1, b2, b3]
    ([b1', b2', b3'], [w1', w2', w3']) = backprop' (P.tail ws) activations zs
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
