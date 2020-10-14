{-# LANGUAGE DataKinds #-}

module Main where

import           Criterion.Main
import           Numeric.LinearAlgebra.Data as LA
import           Numeric.LinearAlgebra.HMatrix ((<.>), mul)
import           System.Random

import qualified Data.Array.Accelerate as Acc
import           Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as NLA

import Data.Array.Accelerate.Control.Lens.Shape
import Control.Lens.Tuple (_2)

main :: IO ()
main = defaultMain
     [ fixedSizeMult 1000    42
     , varSizeMult   1000    39 43
     , dotProd       1000000 234
     , padding       2 1000
     ]

fixedSizeMult :: Int -> Int -> Benchmark
fixedSizeMult n seed = bgroup "Fixed size array multiplication"
  [ bench "hmatrix"
    $ nf (foldl1 mul) (take n $ hmatrixRandFixed 128 0 1 $ mkStdGen seed)
  , bench "accelerate"
    $ nf (CPU.run . foldl1 (NLA.<>)) (take n $ accRandFixed 128 0 1 $ mkStdGen seed)
  ]


varSizeMult :: Int -> Int -> Int -> Benchmark
varSizeMult n seed1 seed2 = bgroup "Fixed size array multiplication"
  [ bench "hmatrix"
    $ nf (foldl1 mul) (take n $ hmatrixRandVar 100 156 (mkStdGen seed1) 0 1 (mkStdGen seed2))
  , bench "accelerate"
    $ nf (CPU.run . foldl1 (NLA.<>)) (take n $ accRandVar 100 156 (mkStdGen seed1) 0 1 (mkStdGen seed2))
  ]

dotProd :: Int -> Int -> Benchmark
dotProd n seed
  = bgroup "Dot Product"
  [ bench "hmatrix"
    $ nf (uncurry (<.>)) (hmatrixRandVec n 0 1 (mkStdGen seed))
  , bench "accelerate"
    $ nf (CPU.run . uncurry (NLA.<.>)) (accRandVec n 0 1 (mkStdGen seed))  
  ]

padding :: Int -> Int -> Benchmark
padding step n
  = bgroup "Padding"
  [ bench "hmatrix"
    $ nf (head . drop n . iterate (hmatrixPad step step)) (konst 10 (5, 5))
  , bench "accelerate"
    $ nf (CPU.run . head . drop n . iterate (accPad (Acc.constant step) (Acc.constant step)))
         (Acc.fill (Acc.index2 (Acc.constant 5) (Acc.constant 5)) (Acc.constant 10))
  ]

{-----------------------}
{-- hmatrix functions --}
{-----------------------}

hmatrixRandFixed :: RandomGen g
                 => Int                   -- ^ Array size
                 -> Double -> Double -> g -- ^ Min and max vals, val rng
                 -> [LA.Matrix Double]    -- ^ Arrays
hmatrixRandFixed size vmin vmax vrng
  = map fromLists $ makeMatrices (repeat size) $ randomRs (vmin, vmax) vrng

hmatrixRandVar :: RandomGen g
               => Int -> Int -> g       -- ^ Min and max size, size rng
               -> Double -> Double -> g -- ^ Min and max vals, val rng
               -> [LA.Matrix Double]    -- ^ Arrays
hmatrixRandVar smin smax srng vmin vmax vrng
  = map fromLists $ makeMatrices (randomRs (smin, smax) srng) $ randomRs (vmin, vmax) vrng

hmatrixRandVec :: RandomGen g
               => Int                            -- ^ Size
               -> Double -> Double -> g          -- ^ Min and max vals, val rng
               -> (LA.Vector Double, LA.Vector Double) -- ^ Vectors
hmatrixRandVec size vmin vmax vrng
  = let (vec1 : vec2 : _) = chunksOf size (randomRs (vmin, vmax) vrng)
    in (LA.fromList vec1, LA.fromList vec2)

hmatrixPad :: Int -> Int -> LA.Matrix Double -> LA.Matrix Double
hmatrixPad row col mat = (mat ||| right) === bottom
  where
    right  = konst 0 (rows mat, col)
    bottom = konst 0 (row, cols mat + col)

makeMatrices :: [Int] -> [a] -> [[[a]]]
makeMatrices (row : col : dims) xs = matrix : makeMatrices (col : dims) rest
  where
    (vals, rest) = splitAt (row * col) xs
    matrix       = chunksOf col vals

{--------------------------}
{-- accelerate functions --}
{--------------------------}


accRandFixed :: RandomGen g
             => Int                       -- ^ Array size
             -> Double -> Double -> g     -- ^ Min and max vals, val rng
             -> [Acc (Acc.Matrix Double)] -- ^ Arrays
accRandFixed size vmin vmax vrng
  = makeMatrices' (repeat size) (randomRs (vmin, vmax) vrng)

accRandVar :: RandomGen g
               => Int -> Int -> g           -- ^ Min and max size, size rng
               -> Double -> Double -> g     -- ^ Min and max vals, val rng
               -> [Acc (Acc.Matrix Double)] -- ^ Arrays
accRandVar smin smax srng vmin vmax vrng
  = makeMatrices' (randomRs (smin, smax) srng) (randomRs (vmin, vmax) vrng)

accRandVec :: RandomGen g
               => Int                                                -- ^ Size
               -> Double -> Double -> g                              -- ^ Min and max vals, val rng
               -> (Acc (Acc.Vector Double), Acc (Acc.Vector Double)) -- ^ Vectors
accRandVec size vmin vmax vrng
  = let (vec1 : vec2 : _) = chunksOf size (randomRs (vmin, vmax) vrng)
    in (Acc.use $ Acc.fromList (Acc.Z Acc.:. size) vec1, Acc.use $ Acc.fromList (Acc.Z Acc.:. size) vec2)

makeMatrices' :: [Int] -> [Double] -> [Acc (Acc.Matrix Double)]
makeMatrices' (row : col : dims) xs = matrix : makeMatrices' (col : dims) rest
  where
    (vals, rest) = splitAt (row * col) xs
    matrix       = Acc.use $ Acc.fromList (Acc.Z Acc.:. row Acc.:. col) vals

accPad :: Acc.Exp Int -> Acc.Exp Int -> Acc (Acc.Matrix Double) -> Acc (Acc.Matrix Double)
accPad row col mat = Acc.concatOn _2 (mat Acc.++ right) bottom
  where
    rc     = Acc.unindex2 (Acc.shape mat)
    right  = Acc.fill (Acc.index2 (Acc.fst rc) col)       (Acc.constant 0)
    bottom = Acc.fill (Acc.index2 row (Acc.snd rc + col)) (Acc.constant 0)

{----------------------}
{-- helper functions --}
{----------------------}

chunksOf :: Int -> [a] -> [[a]]
chunksOf n xs = let go [] = []
                    go ys = take n ys : go (drop n ys)
                 in go xs
