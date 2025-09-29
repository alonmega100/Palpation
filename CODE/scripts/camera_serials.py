import pyrealsense2 as rs

ctx = rs.context()
devs = list(ctx.devices)
if not devs:
    print("No RealSense devices found.")
for d in devs:
    info = {}
    for k in [
        rs.camera_info.name,
        rs.camera_info.serial_number,
        rs.camera_info.firmware_version,
        rs.camera_info.product_line,
        rs.camera_info.usb_type_descriptor,
    ]:
        if d.supports(k):
            info[k] = d.get_info(k)
    sensors = [s.get_info(rs.camera_info.name) for s in d.query_sensors() if s.supports(rs.camera_info.name)]
    print(f"{info.get(rs.camera_info.name, 'Unknown')} | "
          f"SN: {info.get(rs.camera_info.serial_number, '?')} | "
          f"FW: {info.get(rs.camera_info.firmware_version, '?')} | "
          f"Line: {info.get(rs.camera_info.product_line, '?')} | "
          f"USB: {info.get(rs.camera_info.usb_type_descriptor, '?')} | "
          f"Sensors: {', '.join(sensors)}")
