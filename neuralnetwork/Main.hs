module Main where

import Data.Array.Accelerate              as A
import Data.Array.Accelerate.LLVM.Native  as CPU

import qualified Network as AccNet
import qualified Network2 as HNet
import Numeric.LinearAlgebra as NLA

import System.CPUTime
import Text.Printf

main :: IO ()
main = do
  let input    = Prelude.reverse [[1, 0], [0, 1], [1, 1], [0, 0]]
  let expected = Prelude.reverse [[1],    [1],    [0],    [0]   ]
  let xorData = [ (use $ (A.fromList (Z :. 2) x :: A.Vector Double), use $ (A.fromList (Z :. 1) y :: A.Vector Double)) | 
                  (x, y) <- Prelude.zip input expected ]
  let net  = AccNet.create [2, 2, 1]
  let epochs = 100
  putStrLn $ "Begin training for " Prelude.++ show epochs Prelude.++ " epochs (accelerate)"
  start <- getCPUTime
  let net' = AccNet.train net 4 xorData epochs 2
  let feedforward' = CPU.run1 (AccNet.feedforward net')
  putStrLn $ show $ feedforward' (A.fromList (Z :. 2) [0, 0] :: A.Vector Double)
  putStrLn $ show $ feedforward' (A.fromList (Z :. 2) [1, 0] :: A.Vector Double)
  putStrLn $ show $ feedforward' (A.fromList (Z :. 2) [1, 1] :: A.Vector Double)
  putStrLn $ show $ feedforward' (A.fromList (Z :. 2) [0, 1] :: A.Vector Double)
  end   <- getCPUTime
  putStrLn "End"
  let diff = (Prelude.fromIntegral (end - start)) / (10 Prelude.^ 12)
  printf "Accelerate: Computation time: %0.3f sec\n" (diff :: Double)
  
  let xorData = [ (NLA.fromList i, NLA.fromList e) | (i, e) <- Prelude.zip input expected ]
  let net  = HNet.create [2, 2, 1]
  putStrLn $ "Begin training for " Prelude.++ show epochs Prelude.++ " epochs (hmatrix)"
  start <- getCPUTime
  let net' = HNet.train net 4 xorData epochs 2
  putStrLn $ show $ HNet.feedforward net' (NLA.fromList [0, 0])
  putStrLn $ show $ HNet.feedforward net' (NLA.fromList [0, 1])
  putStrLn $ show $ HNet.feedforward net' (NLA.fromList [1, 0])
  putStrLn $ show $ HNet.feedforward net' (NLA.fromList [1, 1])
  end   <- getCPUTime
  putStrLn "End"
  let diff = (Prelude.fromIntegral (end - start)) / (10 Prelude.^ 12)
  printf "HMatrix: Computation time: %0.3f sec\n" (diff :: Double)
