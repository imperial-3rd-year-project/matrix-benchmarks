module Main where

import qualified Data.Array.Accelerate as A

import Data.Array.Accelerate.Data.Colour.RGB (packRGB, unpackRGB, Colour, RGB(..))
import Data.Array.Accelerate.IO.Codec.BMP (readImageFromBMP, writeImageToBMP)
import Data.Array.Accelerate.LLVM.Native (run)

dir, img1, img2, img3 :: String
dir  = "/home/king/soft-eng/array-testing/"
img1 = dir ++ "img1.bmp"
img2 = dir ++ "img2.bmp"
img3 = dir ++ "img3.bmp"
img4 = dir ++ "img4.bmp"

main :: IO ()
main = do
  bg  <- A.map unpackRGB . A.lift . either (error . show) id <$> readImageFromBMP img1
  img <- A.map unpackRGB . A.lift . either (error . show) id <$> readImageFromBMP img2
  bg' <- A.map unpackRGB . A.lift . either (error . show) id <$> readImageFromBMP img3
  let img' = A.zipWith (\b t -> b A.? (A.unlift t)) (A.zipWith (diff 5) bg img) (A.zip img bg')
  writeImageToBMP img4 (run $ A.map packRGB img')

diff :: Float -> A.Exp Colour -> A.Exp Colour -> A.Exp Bool
diff tolerance rgb rgb' = f r r' A.|| f g g' A.|| f b b'
  where
    RGB r  g  b  = A.unlift rgb
    RGB r' g' b' = A.unlift rgb

    f :: A.Exp Float -> A.Exp Float -> A.Exp Bool
    f x y = (max x y - min x y) A.> (A.lift tolerance)
