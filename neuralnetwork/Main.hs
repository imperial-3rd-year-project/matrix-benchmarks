{-# LANGUAGE BangPatterns #-}

module Main where

import Data.Array.Accelerate              as A hiding ((!!))
import Data.Array.Accelerate.LLVM.Native  as CPU

import Network

main :: IO ()
main = do
  -- putStrLn $ show $ CPU.run $ dotp (use xs) (use ys)
  let input    = Prelude.reverse [[1, 0], [0, 1], [1, 1], [0, 0]]
  let expected = Prelude.reverse [[1],    [1],    [0],    [0]   ]
  let xorData = [ (use $ (fromList (Z :. 2) x :: Vector Double), use $ (fromList (Z :. 1) y :: Vector Double)) | (x, y) <- Prelude.zip input expected ]
  let net  = create [2, 2, 1]
  putStrLn "Begin"
  let net' = train net 4 xorData 500 2
  putStrLn $ show $ CPU.run $ feedforward net' (use $ (fromList (Z :. 2) [0, 0] :: Vector Double))
  putStrLn $ show $ CPU.run $ feedforward net' (use $ (fromList (Z :. 2) [1, 0] :: Vector Double))
  putStrLn $ show $ CPU.run $ feedforward net' (use $ (fromList (Z :. 2) [1, 1] :: Vector Double))
  putStrLn $ show $ CPU.run $ feedforward net' (use $ (fromList (Z :. 2) [0, 1] :: Vector Double))
  putStrLn "End"


{--
import Network2
import Numeric.LinearAlgebra as NLA

main :: IO ()
main = do
  -- putStrLn $ show $ CPU.run $ dotp (use xs) (use ys)
  let input    = reverse [[1, 0], [0, 1], [1, 1], [0, 0]]
  let expected = reverse [[1],    [1],    [0],    [0]   ]
  let xorData = [ (fromList i, fromList e) | (i, e) <- zip input expected ]
  let net  = create [2, 2, 1]
  putStrLn "Training begins"
  let net' = train net 4 xorData 500 2
  putStrLn $ show $ feedforward net' (fromList [0, 0])
  putStrLn "Training ends"
  putStrLn $ show $ feedforward net' (fromList [0, 1])
  putStrLn $ show $ feedforward net' (fromList [1, 0])
  putStrLn $ show $ feedforward net' (fromList [1, 1])

--}