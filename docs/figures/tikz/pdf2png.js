// Render page 1 of a PDF to a PNG at a given pixel width, through macOS PDFKit/AppKit (vector-accurate, no extra tools).
//   osascript -l JavaScript pdf2png.js in.pdf out.png 3000
ObjC.import("AppKit");
function run(argv) {
  var inPath = argv[0], outPath = argv[1], widthPx = parseInt(argv[2] || "3000", 10);
  var data = $.NSData.dataWithContentsOfFile(inPath);
  var img = $.NSImage.alloc.initWithData(data);
  var size = img.size;                                    // points
  var scale = widthPx / size.width;
  var w = Math.round(size.width * scale), h = Math.round(size.height * scale);
  var rep = $.NSBitmapImageRep.alloc.initWithBitmapDataPlanesPixelsWidePixelsHighBitsPerSampleSamplesPerPixelHasAlphaIsPlanarColorSpaceNameBytesPerRowBitsPerPixel(
    null, w, h, 8, 4, true, false, $.NSCalibratedRGBColorSpace, 0, 0);
  rep.setSize($.NSMakeSize(size.width, size.height));    // keep the point size so drawing maps points -> pixels at `scale`
  $.NSGraphicsContext.saveGraphicsState;
  var ctx = $.NSGraphicsContext.graphicsContextWithBitmapImageRep(rep);
  $.NSGraphicsContext.setCurrentContext(ctx);
  ctx.setImageInterpolation($.NSImageInterpolationHigh);
  $.NSColor.whiteColor.setFill;
  $.NSBezierPath.fillRect($.NSMakeRect(0, 0, size.width, size.height));
  img.drawInRectFromRectOperationFraction($.NSMakeRect(0, 0, size.width, size.height), $.NSZeroRect, $.NSCompositingOperationSourceOver, 1.0);
  $.NSGraphicsContext.restoreGraphicsState;
  var png = rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $.NSDictionary.dictionary);
  png.writeToFileAtomically(outPath, true);
  return inPath + " -> " + outPath + " " + w + "x" + h + " px (" + Math.round(72 * scale) + " dpi)";
}
