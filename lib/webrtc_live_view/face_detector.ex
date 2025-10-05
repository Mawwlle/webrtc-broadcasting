defmodule WebRTCLiveView.FaceDetector do
  use Membrane.Filter

  def_input_pad :input, accepted_format: %Membrane.RawVideo{pixel_format: :RGB}
  def_output_pad :output, accepted_format: %Membrane.RawVideo{pixel_format: :RGB}

  @output_width 640
  @output_height 480

  @impl true
  def handle_init(_ctx, _opts) do
    face_cascade_path =
      Path.join([
        :code.priv_dir(:evision),
        "share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
      ])

    face_cascade = Evision.CascadeClassifier.cascadeClassifier(face_cascade_path)
    {[], %{face_cascade: face_cascade}}
  end

  @impl true
  def handle_buffer(:input, buffer, ctx, state) do
    %{height: height, width: width} = ctx.pads.input.stream_format

    {:ok, vips_image} =
      Vix.Vips.Image.new_from_binary(buffer.payload, width, height, 3, :VIPS_FORMAT_UCHAR)

    {:ok, evision_image} = Image.to_evision(vips_image)

    grey_img = Evision.cvtColor(evision_image, Evision.Constant.cv_COLOR_RGB2GRAY())

    faces =
      Evision.CascadeClassifier.detectMultiScale(
        state.face_cascade,
        grey_img
      )

    largest_face =
      if Enum.empty?(faces) do
        # No face detected - use whole image
        {0, 0, width, height}
      else
        # Find largest face
        Enum.reduce(faces, nil, fn cur_face = {_, _, w, h}, acc ->
          cur_area = w * h

          case acc do
            nil ->
              cur_face

            {_, _, acc_w, acc_h} ->
              acc_area = acc_w * acc_h
              if cur_area > acc_area, do: cur_face, else: acc
          end
        end)
      end

    {x, y, w, h} = largest_face

    cropped_evision = Evision.Mat.roi(evision_image, {x, y, w, h})

    # Resize to consistent output dimensions
    resized_evision = Evision.resize(cropped_evision, {@output_width, @output_height})

    {:ok, resized_image} = Image.from_evision(resized_evision)

    {:ok, payload} =
      resized_image
      |> Image.flatten!()
      |> Image.to_colorspace!(:srgb)
      |> Vix.Vips.Image.write_to_binary()

    buffer = %Membrane.Buffer{buffer | payload: payload}
    {[buffer: {:output, buffer}], state}
  end
end
