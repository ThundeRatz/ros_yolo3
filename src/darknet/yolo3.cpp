/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 ThundeRatz

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "darknet/yolo3.h"

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <yolo3/Detection.h>
#include <yolo3/ImageDetections.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

extern "C"
{
#undef __cplusplus
#include "detection_layer.h"  // NOLINT(build/include)
#include "parser.h"  // NOLINT(build/include)
#include "region_layer.h"  // NOLINT(build/include)
#include "utils.h"  // NOLINT(build/include)
detection *make_network_boxes(network *net, float thresh, int *num);
#define __cplusplus
}

namespace darknet
{
void Detector::load(std::string& model_file, std::string& trained_file, double min_confidence, double nms)
{
  min_confidence_ = min_confidence;
  nms_ = nms;
  net_ = parse_network_cfg(&model_file[0]);
  load_weights(net_, &trained_file[0]);
  set_batch_network(net_, 1);

  detections_ = make_network_boxes(net_, min_confidence_, &number_of_boxes_);
}

Detector::~Detector()
{
  free_network(net_);
  for (int i = 0; i < number_of_boxes_; i++)
  {
    free(detections_[i].prob);
  }
  free(detections_);
}

yolo3::ImageDetections Detector::detect(float *data, int original_width, int original_height)
{
  yolo3::ImageDetections detections;
  detections.detections = forward(data, original_width, original_height);
  return detections;
}

image Detector::convert_image(const sensor_msgs::ImageConstPtr& msg)
{
  if (msg->encoding != sensor_msgs::image_encodings::RGB8)
  {
    ROS_ERROR("Unsupported encoding");
    exit(-1);
  }

  auto data = msg->data;
  uint32_t height = msg->height, width = msg->width, offset = msg->step - 3 * width;
  uint32_t i = 0, j = 0;
  image im = make_image(width, height, 3);

  for (uint32_t line = height; line; line--)
  {
    for (uint32_t column = width; column; column--)
    {
      for (uint32_t channel = 0; channel < 3; channel++)
        im.data[i + width * height * channel] = data[j++] / 255.;
      i++;
    }
    j += offset;
  }

  if (net_->w == static_cast<int>(width) && net_->h == static_cast<int>(height))
  {
    return im;
  }
  image resized = letterbox_image(im, net_->w, net_->h);
  free_image(im);
  return resized;
}

std::vector<yolo3::Detection> Detector::forward(float *data, int original_width, int original_height)
{
  network_predict(net_, data);
  layer output_layer = net_->layers[net_->n - 1];

  int count = get_yolo_detections(output_layer, original_width, original_height, net_->w, net_->h, min_confidence_, 0, 1, detections_);

  int num_classes = output_layer.classes;
  if (nms_)
    do_nms_sort(detections_, number_of_boxes_, num_classes, nms_);
  std::vector<yolo3::Detection> detections;
  for (int i = 0; i < count; i++)
  {
    int class_id = max_index(detections_[i].prob, num_classes);
    float prob = detections_[i].prob[class_id];
    if (prob)
    {
      yolo3::Detection detection;
      box b = detections_[i].bbox;

      detection.x = b.x;
      detection.y = b.y;
      detection.width = b.w;
      detection.height = b.h;
      detection.confidence = prob;
      detection.class_id = class_id;
      detections.push_back(detection);
    }
  }
  return detections;
}
}  // namespace darknet
