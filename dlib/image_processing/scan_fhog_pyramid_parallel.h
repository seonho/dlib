#ifndef DLIB_SCAN_fHOG_PYRAMID_PARALLEL_Hh_
#define DLIB_SCAN_fHOG_PYRAMID_PARALLEL_Hh_

namespace dlib
{
	namespace impl
	{
		template <typename fhog_filterbank>
        rectangle apply_filters_to_fhog_parallel (
            const fhog_filterbank& w,
            const array<array2d<float> >& feats,
            array2d<float>& saliency_image
        )
        {
            const unsigned long num_separable_filters = w.num_separable_filters();
            rectangle area;
            // use the separable filters if they would be faster than running the regular filters.
            if (num_separable_filters > w.filters.size()*std::min(w.filters[0].nr(),w.filters[0].nc())/3.0)
            {
                area = spatially_filter_image(feats[0], saliency_image, w.filters[0]);
                for (unsigned long i = 1; i < w.filters.size(); ++i)
                {
                    // now we filter but the output adds to saliency_image rather than
                    // overwriting it.
                    spatially_filter_image(feats[i], saliency_image, w.filters[i], 1, false, true);
                }
            }
            else
            {
                // find the first filter to apply
				typedef size_t size_type;

                size_type i = 0;
                while (i < w.row_filters.size() && w.row_filters[i].size() == 0) 
                    ++i;

                unsigned long num_filters = 0;
                std::vector<unsigned long> filters_before;
                filters_before.push_back(0);
                for (size_type j = i; j < w.row_filters.size(); ++j)
                {
                	num_filters += w.row_filters[j].size();
                	filters_before.push_back(num_filters);
                }

                array<array2d<float> > saliency_images;
                saliency_images.set_max_size(num_filters);
                saliency_images.set_size(num_filters);

                concurrency_compat::parallel_for(i, w.row_filters.size(), [&](size_type k)
                {
                	array2d<float> tmp_saliency_image;
                	array2d<float> scratch;
                    for (unsigned long j = 0; j < w.row_filters[k].size(); ++j)
                    {
                        area = float_spatially_filter_image_separable(feats[k], tmp_saliency_image, w.row_filters[k][j], w.col_filters[k][j], scratch, false);
                        swap(tmp_saliency_image, saliency_images[filters_before[k] - i + j]);
                    }
                }
                );

                saliency_image.clear();
                saliency_image.set_size(feats[0].nr(), feats[0].nc());
                assign_all_pixels(saliency_image, 0);
                
                // sum across the saliency images
                concurrency_compat::parallel_for(static_cast<long>(0), saliency_image.nr(), [&](long y)
                {
                	for (unsigned long k = 0; k < saliency_images.size(); ++k)
                	{
						//for (long y = 0; y < saliency_image.nr(); ++y)
						{
							for (long x = 0; x < saliency_image.nc(); ++x)
							{
								saliency_image[y][x] += saliency_images[k][y][x];
							}
						}
                	}
                }
                );
            }
            return area;
        }

        template <
            typename pyramid_type,
            typename image_type,
            typename feature_extractor_type
            >
        void create_fhog_pyramid_parallel (
            const image_type& img,
            const feature_extractor_type& fe,
            array<array<array2d<float> > >& feats,
            int cell_size,
            int filter_rows_padding,
            int filter_cols_padding,
            unsigned long min_pyramid_layer_width,
            unsigned long min_pyramid_layer_height,
            unsigned long max_pyramid_levels
        )
        {
            unsigned long levels = 0;
            rectangle rect = get_rect(img);

            // figure out how many pyramid levels we should be using based on the image size
            pyramid_type pyr;
            do
            {
                rect = pyr.rect_down(rect);
                ++levels;
            } while (rect.width() >= min_pyramid_layer_width && rect.height() >= min_pyramid_layer_height &&
                levels < max_pyramid_levels);

            if (feats.max_size() < levels)
                feats.set_max_size(levels);
            feats.set_size(levels);



            // build our feature pyramid
            fe(img, feats[0], cell_size,filter_rows_padding,filter_cols_padding);
            DLIB_ASSERT(feats[0].size() == fe.get_num_planes(), 
                "Invalid feature extractor used with dlib::scan_fhog_pyramid.  The output does not have the \n"
                "indicated number of planes.");

            if (feats.size() > 1)
            {
                typedef typename image_traits<image_type>::pixel_type pixel_type;

                // create image pyramid
                array<array2d<pixel_type>> image_pyramid;
                image_pyramid.set_max_size(feats.size() - 1);
                image_pyramid.set_size(feats.size() - 1);

				pyr(img, image_pyramid[0]);
                for (unsigned long i = 1; i < feats.size() - 1; ++i)
                {
                	pyr(image_pyramid[i - 1], image_pyramid[i]);
                }

                concurrency_compat::parallel_for(static_cast<unsigned long>(0), feats.size(), [&](unsigned long i)
            	{
            		if (i == 0)
            			fe(img, feats[i], cell_size,filter_rows_padding,filter_cols_padding);
            		else
            			fe(image_pyramid[i - 1], feats[i], cell_size,filter_rows_padding,filter_cols_padding);
            	});
            }
        }

		template <
			typename pyramid_type,
			typename feature_extractor_type,
			typename fhog_filterbank
		>
        void detect_from_fhog_pyramid_parallel(
			const array<array<array2d<float> > >& feats,
			const feature_extractor_type& fe,
			const fhog_filterbank& w,
			const double thresh,
			const unsigned long det_box_height,
			const unsigned long det_box_width,
			const int cell_size,
			const int filter_rows_padding,
			const int filter_cols_padding,
			std::vector<std::pair<double, rectangle> >& dets
		)
		{
			dets.clear();

			concurrency_compat::concurrent_vector<std::pair<double, rectangle> > dets_conc;

			pyramid_type pyr;

			// for all pyramid levels
			concurrency_compat::parallel_for(static_cast<unsigned long>(0), feats.size(), [&](unsigned long l)
			{
				array2d<float> saliency_image;

				const rectangle area = apply_filters_to_fhog_parallel(w, feats[l], saliency_image);

				// now search the saliency image for any detections
				for (long r = area.top(); r <= area.bottom(); ++r)
				{
					for (long c = area.left(); c <= area.right(); ++c)
					{
						// if we found a detection
						if (saliency_image[r][c] >= thresh)
						{
							rectangle rect = fe.feats_to_image(centered_rect(point(c, r), det_box_width, det_box_height),
								cell_size, filter_rows_padding, filter_cols_padding);
							rect = pyr.rect_up(rect, l);
							dets_conc.push_back(std::make_pair(saliency_image[r][c], rect));
						}
					}
				}
			}
			);
			dets.assign(dets_conc.begin(), dets_conc.end());

			std::sort(dets.rbegin(), dets.rend(), compare_pair_rect);
		}
	}
}

#endif // DLIB_SCAN_fHOG_PYRAMID_PARALLEL_Hh_