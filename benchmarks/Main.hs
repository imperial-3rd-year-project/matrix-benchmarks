{-# LANGUAGE DataKinds #-}
{-# LANGUAGE CPP #-}

module Main where

import           Criterion.Main
import           Numeric.LinearAlgebra.Data as LA
import           Numeric.LinearAlgebra.HMatrix ((<.>), mul)
import           System.Random

-- massiv
import qualified Data.Massiv.Array as MA

import qualified Data.Array.Accelerate as Acc
import           Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as NLA

import Data.Array.Accelerate.Control.Lens.Shape
import Data.Maybe (fromJust)

#define MATSIZE 128
#define MAXMATSIZE 156

main :: IO ()
main = defaultMain
     [ fixedSizeMult 10    42
     , varSizeMult   10    39 43
     , dotProd       1000000 234
     , padding       2 50
     ]

fixedSizeMult :: Int -> Int -> Benchmark
fixedSizeMult n seed = bgroup ("Fixed size matrix multiplication, " ++ show n ++ " " ++ show MATSIZE ++ "x" ++ show MATSIZE ++ " matrices")
  [ bench "hmatrix"
    $ nf (foldl1 mul)                                 (take n $ hmatrixRandFixed MATSIZE 0 1 $ mkStdGen seed)
  , bench "accelerate"
    $ nf (CPU.run . foldl1 (NLA.<>))                  (take n $ accRandFixed     MATSIZE 0 1 $ mkStdGen seed)
  , bench "massiv"
    $ nf (foldl1 (\m1 m2 -> fromJust (m1 MA.|*| m2))) (take n $ massivRandFixed  MATSIZE 0 1 $ mkStdGen seed)
  ]


varSizeMult :: Int -> Int -> Int -> Benchmark
varSizeMult n seed1 seed2 = bgroup ("Variable size matrix multiplication, " ++ show n ++ " nxm matrices, " ++ show MATSIZE ++ " <= n, m <= " ++ show MAXMATSIZE)
  [ bench "hmatrix"
    $ nf (foldl1 mul) (take n $ hmatrixRandVar MATSIZE MAXMATSIZE (mkStdGen seed1) 0 1 (mkStdGen seed2))
  , bench "accelerate"
    $ nf (CPU.run . foldl1 (NLA.<>)) (take n $ accRandVar MATSIZE MAXMATSIZE (mkStdGen seed1) 0 1 (mkStdGen seed2))
  , bench "massiv"
    $ nf (foldl1 (\m1 m2 -> fromJust (m1 MA.|*| m2))) (take n $ massivRandVar MATSIZE MAXMATSIZE (mkStdGen seed1) 0 1 (mkStdGen seed2))
  ]

dotProd :: Int -> Int -> Benchmark
dotProd n seed
  = bgroup ("Dot Product between vectors of length " ++ show n)
  [ bench "hmatrix"
    $ nf (uncurry (<.>)) (hmatrixRandVec n 0 1 (mkStdGen seed))
  , bench "accelerate"
    $ nf (CPU.run . uncurry (NLA.<.>)) (accRandVec n 0 1 (mkStdGen seed))
  , bench "massiv"
    $ nf (MA.sum  . uncurry (MA.zipWith (*))) (massivRandVec n 0 1 (mkStdGen seed))
  ] 

padding :: Int -> Int -> Benchmark
padding step n
  = bgroup ("Padding matrices starting at size 5x5 in steps of " ++ show step ++ " upto " ++ show n)
  [ bench "hmatrix"
    $ nf (head . drop n . iterate (hmatrixPad step step)) (konst 10 (5, 5))
  , bench "accelerate"
    $ nf (head . drop n . iterate (CPU.run1 (accPad step step))) (CPU.run accmat)
  , bench "massiv"
    $ nf (head . drop n . iterate (massivPad step step)) (MA.compute (MA.replicate MA.Par (MA.Sz2 5 5) 10))
  ]
    where
      accmat = Acc.fill (Acc.index2 (Acc.constant 5) (Acc.constant 5)) (Acc.constant 10)


{----------------------}
{-- massiv functions --}
{----------------------}

massivRandFixed :: RandomGen g
                 => Int                      -- ^ Array size
                 -> Double -> Double -> g    -- ^ Min and max vals, val rng
                 -> [MA.Array MA.P MA.Ix2 Double] -- ^ Arrays
massivRandFixed size vmin vmax vrng
  = map (MA.fromLists' MA.Par) $ makeMatrices (repeat size) $ randomRs (vmin, vmax) vrng

massivRandVar :: RandomGen g
               => Int -> Int -> g       -- ^ Min and max size, size rng
               -> Double -> Double -> g -- ^ Min and max vals, val rng
               -> [MA.Array MA.P MA.Ix2 Double]    -- ^ Arrays
massivRandVar smin smax srng vmin vmax vrng
  = map (MA.fromLists' MA.Par) $ makeMatrices (randomRs (smin, smax) srng) $ randomRs (vmin, vmax) vrng

massivRandVec :: RandomGen g
               => Int                            -- ^ Size
               -> Double -> Double -> g          -- ^ Min and max vals, val rng
               -> (MA.Array MA.P MA.Ix1 Double, MA.Array MA.P MA.Ix1 Double) -- ^ Vectors
massivRandVec size vmin vmax vrng
  = let (vec1 : vec2 : _) = chunksOf size (randomRs (vmin, vmax) vrng)
    in (MA.fromList MA.Par vec1, MA.fromList MA.Par vec2)
    
massivPad :: Int -> Int -> MA.Array MA.P MA.Ix2 Double -> MA.Array MA.P MA.Ix2 Double
massivPad row col mat = MA.compute (MA.concat' (MA.Dim 2) [MA.compute (MA.concat' (MA.Dim 1) [mat, right]), bottom])
  where
    (MA.Sz (mrow MA.:. mcol)) = MA.size mat
    right :: MA.Array MA.P MA.Ix2 Double
    right  = MA.compute (MA.replicate MA.Par (MA.Sz2 mrow col) 0)
    bottom :: MA.Array MA.P MA.Ix2 Double
    bottom = MA.compute (MA.replicate MA.Par (MA.Sz2 row (mcol + col)) 0)
{--
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
--}

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
    in ( Acc.use $ Acc.fromList (Acc.Z Acc.:. size) vec1
       , Acc.use $ Acc.fromList (Acc.Z Acc.:. size) vec2 )

makeMatrices' :: [Int] -> [Double] -> [Acc (Acc.Matrix Double)]
makeMatrices' (row : col : dims) xs = matrix : makeMatrices' (col : dims) rest
  where
    (vals, rest) = splitAt (row * col) xs
    matrix       = Acc.use $ Acc.fromList (Acc.Z Acc.:. row Acc.:. col) vals

accPad :: Int -> Int -> Acc (Acc.Matrix Double) -> Acc (Acc.Matrix Double)
accPad row col mat = Acc.concatOn _2 (mat Acc.++ right) bottom
  where
    row'   = Acc.constant row
    col'   = Acc.constant col
    rc     = Acc.unindex2 (Acc.shape mat)
    right  = Acc.fill (Acc.index2 (Acc.fst rc) col')        (Acc.constant 0)
    bottom = Acc.fill (Acc.index2 row' (Acc.snd rc + col')) (Acc.constant 0)

{----------------------}
{-- helper functions --}
{----------------------}

chunksOf :: Int -> [a] -> [[a]]
chunksOf n xs = let go [] = []
                    go ys = take n ys : go (drop n ys)
                 in go xs
