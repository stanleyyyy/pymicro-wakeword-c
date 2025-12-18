SUMMARY = "Wake word detection using TensorFlow Lite models"
DESCRIPTION = "Pure C/C++ library that provides wake word detection \
using TensorFlow Lite models. Provides a clean C API for processing \
16kHz 16-bit audio and detecting wake words with configurable \
probability thresholds and sliding windows."
HOMEPAGE = "https://github.com/OHF-Voice/pymicro-wakeword"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=86d3f3a95c324c9479bd8986968f4327"

SRC_URI = "git://github.com/stanleyyyy/pymicro-wakeword-c.git;protocol=https;branch=main"
SRCREV = "${AUTOREV}"

S = "${WORKDIR}/git"

# Build flags - Makefile will handle includes and linking
# Override micro-features paths to use installed version from sysroot
EXTRA_OEMAKE = " \
	CC='${CC}' \
	CXX='${CXX}' \
	CFLAGS='${CFLAGS}' \
	CXXFLAGS='${CXXFLAGS}' \
	LDFLAGS='${LDFLAGS}' \
	MICRO_FEATURES_DIR='${STAGING_LIBDIR}' \
	MICRO_FEATURES_INCLUDE='${STAGING_INCDIR}' \
	MICRO_FEATURES_LIB='${STAGING_LIBDIR}/libmicro_features.a' \
"

# Configure step - clean build directory
do_configure() {
	cd ${S}
	oe_runmake -f Makefile.lib clean
}

do_compile() {
	# Build library and all binaries using Makefile
	cd ${S}
	oe_runmake -f Makefile.lib all test
}

do_install() {
	# Install header files
	install -d ${D}${includedir}
	install -m 0644 ${S}/include/micro_wakeword.h ${D}${includedir}/

	# Install static library
	install -d ${D}${libdir}
	install -m 0644 ${S}/libmicro_wakeword.a ${D}${libdir}/

	# Detect architecture and install appropriate TensorFlow Lite library
	# Map bitbake TARGET_ARCH/TUNE_ARCH to lib directory names
	LIB_ARCH=""
	case "${TARGET_ARCH}" in
		arm)
			# For 32-bit ARM, check TUNE_ARCH to distinguish armv7
			case "${TUNE_ARCH}" in
				armv7a|armv7ve|armv7|armv6|armv5te)
					LIB_ARCH="linux_armv7"
					;;
				*)
					# Default to armv7 for 32-bit arm
					LIB_ARCH="linux_armv7"
					;;
			esac
			;;
		aarch64)
			LIB_ARCH="linux_arm64"
			;;
		x86_64)
			LIB_ARCH="linux_amd64"
			;;
		*)
			bbwarn "Unknown architecture ${TARGET_ARCH}, defaulting to linux_amd64"
			LIB_ARCH="linux_amd64"
			;;
	esac

	# Install TensorFlow Lite dynamic library
	if [ -f "${S}/lib/${LIB_ARCH}/libtensorflowlite_c.so" ]; then
		install -m 0755 ${S}/lib/${LIB_ARCH}/libtensorflowlite_c.so ${D}${libdir}/
	else
		bbfatal "TensorFlow Lite library not found for architecture ${LIB_ARCH} at ${S}/lib/${LIB_ARCH}/libtensorflowlite_c.so"
	fi

	# Install example binaries
	install -d ${D}${bindir}
	install -m 0755 ${S}/examples/example_c ${D}${bindir}/
	install -m 0755 ${S}/examples/example_cpp ${D}${bindir}/

	# Install test binary (optional, can be removed if not needed)
	install -m 0755 ${S}/tests/test_micro_wakeword ${D}${bindir}/

	# Install model files (.tflite and .json) to a relative location
	# This allows tests to find them via ./models/ path
	install -d ${D}${bindir}/models
	for model_file in ${S}/pymicro_wakeword/models/*.tflite ${S}/pymicro_wakeword/models/*.json; do
		if [ -f "$model_file" ]; then
			install -m 0644 "$model_file" ${D}${bindir}/models/
		fi
	done

	# Install test WAV files if they exist (tests/{model_name}/*.wav)
	# Install to relative location so tests can find them
	if [ -d "${S}/tests" ]; then
		for model_dir in ${S}/tests/*/; do
			if [ -d "$model_dir" ]; then
				model_name=$(basename "$model_dir")
				# Check if this directory contains .wav files (likely a test model directory)
				if ls "$model_dir"/*.wav 1> /dev/null 2>&1; then
					install -d ${D}${bindir}/tests/${model_name}
					install -m 0644 "$model_dir"/*.wav ${D}${bindir}/tests/${model_name}/ 2>/dev/null || true
				fi
			fi
		done
	fi
}

# Package configuration
PACKAGES = "${PN} ${PN}-dev ${PN}-staticdev ${PN}-examples ${PN}-tests ${PN}-models ${PN}-dbg"

FILES:${PN} = "${libdir}/libtensorflowlite_c.so"
FILES:${PN}-dev = "${includedir}/micro_wakeword.h"
FILES:${PN}-staticdev = "${libdir}/libmicro_wakeword.a"
FILES:${PN}-examples = "${bindir}/example_c ${bindir}/example_cpp"
FILES:${PN}-tests = "${bindir}/test_micro_wakeword ${bindir}/tests ${bindir}/models"
FILES:${PN}-models = "${bindir}/models"
FILES:${PN}-dbg = "${bindir}/.debug"

# Dependencies - requires micro-features-c library
DEPENDS = "micro-features-c"

RDEPENDS:${PN} = ""
RDEPENDS:${PN}-staticdev = "micro-features-c-staticdev"
RDEPENDS:${PN}-examples = "${PN} micro-features-c"
RDEPENDS:${PN}-tests = "${PN} ${PN}-models"

# Skip QA check for already-stripped TensorFlow Lite library
# The pre-built library is already stripped, which is expected
INSANE_SKIP:${PN} += "already-stripped"
