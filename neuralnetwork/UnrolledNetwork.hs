{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeOperators #-}

module UnrolledNetwork where

import Data.Array.Accelerate              as A
import Data.Array.Accelerate.Numeric.LinearAlgebra hiding (trace)
import Prelude as P hiding ((==), (!!))
import qualified Prelude as P
import Data.Array.Accelerate.LLVM.Native as CPU

-- Utilities

-- evalNet net = let BasicNetwork ws bs = net
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
type BasicNetwork = (Layer, Layer)

create :: [Int] -> Acc BasicNetwork
create xs = lift (lift (w1, b1) :: Acc Layer, lift (w2, b2) :: Acc Layer)
  where
    [w1, w2] = do
      idx <- [1..(P.length xs - 1)]
      pure $ use $ (fromList ( Z :. xs P.!! idx :. xs P.!! (idx - 1) ) [1..] :: Matrix Double)
    
    [b1, b2] = do
      idx <- [1..(P.length xs - 1)]
      pure $ use $ (fromList (Z :. xs P.!! idx) [1..] :: Vector Double)


feedforward :: Acc BasicNetwork -> Acc (Vector Double) -> Acc (Vector Double)
feedforward (unlift -> (layer1 :: Acc Layer, layer2 :: Acc Layer)) x = feedLayer layer2 (feedLayer layer1 x)
  where
    feedLayer :: Acc Layer -> Acc (Vector Double) -> Acc (Vector Double)
    feedLayer (unlift -> (w :: Acc (Matrix Double), b :: Acc (Vector Double))) x =
      A.map sigmoid $ (w #> x) ^+^ b

type TrainingData = (Matrix Double, Matrix Double)

trainOnce :: Int -> Double -> Acc TrainingData -> Acc BasicNetwork -> Acc BasicNetwork
trainOnce n eta (unlift -> (xs :: Acc (Matrix Double), ys :: Acc (Matrix Double)))
          net@(unlift -> (layer1 :: Acc Layer, layer2 :: Acc Layer)) = lift (layer1', layer2')
  where
    Z :. len :. _ = unlift (shape xs) :: Z :. Exp Int :. Exp Int
    descend :: Acc (Scalar Int) -> Acc BasicNetwork -> Acc (Scalar Int, BasicNetwork)
    descend (the -> ix) (unlift -> (layer1 :: Acc Layer, layer2 :: Acc Layer)) =
      let (w1, b1) = unlift layer1
          (w2, b2) = unlift layer2
          sh = lift (Z :. ix :. All)
          x = slice xs sh
          y = slice ys sh
          (layer1' :: Acc Layer, layer2' :: Acc Layer) = unlift (backprop x y net)
          (w1', b1') = unlift layer1'
          (w2', b2') = unlift layer2'
          layer1'' :: Acc Layer = lift (w1 ^+^ w1', b1 ^+^ b1')
          layer2'' :: Acc Layer = lift (w2 ^+^ w2', b2 ^+^ b2')
       in lift (unit (ix + 1), lift (layer1'', layer2'') :: Acc BasicNetwork)

    (w1 :: Acc (Matrix Double), b1 :: Acc (Vector Double)) = unlift layer1
    (w2 :: Acc (Matrix Double), b2 :: Acc (Vector Double)) = unlift layer2

    w1Shape = A.shape w1
    b1Shape = A.shape b1
    w2Shape = A.shape w2
    b2Shape = A.shape b2

    w1Zero = A.fill w1Shape 0
    b1Zero = A.fill b1Shape 0
    w2Zero = A.fill w2Shape 0
    b2Zero = A.fill b2Shape 0

    zeroLayer1 :: Acc Layer = lift (w1Zero, b1Zero)
    zeroLayer2 :: Acc Layer = lift (w2Zero, b2Zero)

    zeroNetwork :: Acc BasicNetwork = lift (zeroLayer1, zeroLayer2)

    (nabla1 :: Acc Layer, nabla2 :: Acc Layer) = unlift $ A.asnd $ unlift $
      awhile (A.uncurry (\(the -> ix) _ -> unit (ix A.< len)))
        (A.uncurry descend)
        (lift (unit 0, zeroNetwork))

    velocity = constant (eta / P.fromIntegral n)
    (nablaW1, nablaB1) = unlift nabla1
    (nablaW2, nablaB2) = unlift nabla2

    layer1' :: Acc Layer = lift (w1 ^-^ (velocity *^ nablaW1), b1 ^-^ (velocity *^ nablaB1))
    layer2' :: Acc Layer = lift (w2 ^-^ (velocity *^ nablaW2), b2 ^-^ (velocity *^ nablaB2))
        

train :: Int -> Int -> Double -> Acc TrainingData -> Acc BasicNetwork -> Acc BasicNetwork
train (constant -> epochs) n eta td net = A.asnd $
  awhile (A.uncurry (\(the -> ix) _ -> unit (ix A.< epochs)))
    (A.uncurry (\(the -> ix) net -> lift (unit $ ix + 1, train' td net))) (lift (unit 0, net))
  where
    train' = trainOnce n eta

backprop :: Acc (Vector Double)
         -> Acc (Vector Double)
         -> Acc BasicNetwork
         -> Acc BasicNetwork
backprop actual expected net = lift (lift (y, delta') :: Acc Layer, lift (nw, delta) :: Acc Layer)
  where
    (layer1, layer2) = unlift net
    (w2 :: Acc (Matrix Double), b2 :: Acc (Vector Double)) = unlift layer2

    delta = (cost' a3 expected) ^*^ (A.map sigmoid' z2)
    nw = delta >< a2

    sp     = A.map sigmoid' z1
    delta' = (transpose w2 #> delta) ^*^ sp
    y      = delta' >< a1


    activate :: Acc (Vector Double) -> Acc Layer -> Acc (Vector Double)
    activate x layer = (w #> x) ^+^ b
      where
        (w, b) = unlift layer

    z1, z2 :: Acc (Vector Double)
    z1 = activate actual layer1
    z2 = activate a2 layer2

    a1 = actual
    a2 = A.map sigmoid z1
    a3 = A.map sigmoid z2

cost' :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
cost' actual expected = actual ^-^ expected
