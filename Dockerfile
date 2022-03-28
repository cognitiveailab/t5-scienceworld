FROM python:3.9

CMD ["/bin/bash"]

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
   &&  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs


ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
   &&  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 5001 8883 8888 9000
EXPOSE 25300-25600
USER root:root
WORKDIR /opt
RUN apt-get update \
   &&  apt-get install -y --no-install-recommends default-jre
RUN apt-get install -y --no-install-recommends unzip
EXPOSE 5002-5100
EXPOSE 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011 5012 5013 5014 5015 5016 5017 5018
EXPOSE 50022 50023 50024 50025 50026 50027 50028 50029 50030 50031 50032 50034 50035 50036 50037 50038 50039
EXPOSE 50022 50032 50042 50052 50062 50072 50082 50092 50102 50112 50122 50132 50142 50152 50162 50172 50182
EXPOSE 6379
COPY . /t5-scienceworld
RUN pip install -r /t5-scienceworld/requirements.txt

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y

WORKDIR /

ENV PYTHONPATH=/t5-scienceworld
ENV HOME=""
WORKDIR /t5-scienceworld