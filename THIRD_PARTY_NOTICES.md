# Third-Party Notices

ScaleResfusion is licensed under the PolyForm Noncommercial License 1.0.0 (see [`LICENSE`](LICENSE)). The method it implements is protected by Chinese Invention Patent ZL 2026 1 1122958.7 (CN 122636463 B; Patent Certificate No. 9331537), owned by Nankai University.

This repository also contains code written by third parties. **Those portions are not licensed under the PolyForm Noncommercial License by the ScaleResfusion licensor; they remain under their original licenses**, listed below and reproduced in full at the end of this file. Where a file combines third-party code with ScaleResfusion modifications, the third-party portions remain under the original license and the ScaleResfusion modifications are licensed under [`LICENSE`](LICENSE).

A commercial license from the ScaleResfusion licensor covers only the ScaleResfusion code and the patent rights held by the patentee. It does **not** grant rights to third-party code beyond what that code's own license allows. In particular, see the **non-commercial restriction** in section 3.

## 1. Summary

| Path in this repository | Origin | License | Commercial use of this component |
|---|---|---|---|
| `ram/models/ram.py`, `ram/models/tag2text.py`, `ram/models/utils.py`, `ram/inference.py`, `ram/transform.py`, `ram/__init__.py`, `ram/models/__init__.py`, `ram/configs/**`, `ram/data/**` | [Recognize Anything (RAM / Tag2Text)](https://github.com/xinyu1205/recognize-anything), Copyright (c) 2022 OPPO | Apache License 2.0 | Permitted by the component license |
| `ram/models/ram_lora.py`, `ram/models/tag2text_lora.py`, `ram/models/bert_lora.py`, `ram/models/swin_transformer_lora.py` | LoRA variants used by DAPE, from [SeeSR](https://github.com/cswry/SeeSR) / [OSEDiff](https://github.com/cswry/OSEDiff), based on the RAM, BLIP and Swin Transformer files below | Apache License 2.0, plus the licenses of the underlying files (BSD-3-Clause for `bert_lora.py`, MIT for `swin_transformer_lora.py`) | Permitted by the component licenses |
| `ram/models/bert.py`, `ram/models/bert_lora.py` | [BLIP](https://github.com/salesforce/BLIP), Copyright (c) 2022 Salesforce.com, Inc.; based on [🤗 Transformers](https://github.com/huggingface/transformers) | BSD 3-Clause (BLIP); Apache License 2.0 (Transformers) | Permitted by the component licenses |
| `ram/models/vit.py` | [BLIP](https://github.com/salesforce/BLIP), Copyright (c) 2022 Salesforce.com, Inc.; based on [timm](https://github.com/huggingface/pytorch-image-models) | BSD 3-Clause (BLIP); Apache License 2.0 (timm) | Permitted by the component licenses |
| `ram/models/swin_transformer.py`, `ram/models/swin_transformer_lora.py` | [Swin Transformer](https://github.com/microsoft/Swin-Transformer), Copyright (c) Microsoft Corporation | MIT License | Permitted by the component license |
| `ram/models/utils.py` (asymmetric loss, marked "copy from ASL") | [ASL](https://github.com/Alibaba-MIIL/ASL), Copyright (c) 2020 Alibaba-MIIL | MIT License | Permitted by the component license |
| `scheduler/scheduling_flow_match_euler_discrete.py` (upstream portions) | [🤗 Diffusers](https://github.com/huggingface/diffusers) `FlowMatchEulerDiscreteScheduler`, Copyright The HuggingFace Team | Apache License 2.0 | Upstream portions: permitted by the component license. ScaleResfusion / Resfusion modifications: see [`LICENSE`](LICENSE) |
| `stableresfusion_utils/flux2.py` (functions marked "Copied from diffusers") | [🤗 Diffusers](https://github.com/huggingface/diffusers), Copyright The HuggingFace Team | Apache License 2.0 | Upstream portions: permitted by the component license |
| `my_utils/wavelet_color_fix.py` | Colour-fix script by Li Yi, [sd-webui-stablesr](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py), derived from [StableSR](https://github.com/IceClear/StableSR), Copyright 2022 S-Lab | **S-Lab License 1.0 (non-commercial only)** | **Not permitted** without permission from the S-Lab copyright holder(s); see section 3 |

All other files in this repository are original ScaleResfusion code and are licensed under [`LICENSE`](LICENSE).

## 2. Apache License 2.0 components

The components marked "Apache License 2.0" above are redistributed under the Apache License, Version 2.0 (full text in [Appendix A](#appendix-a-apache-license-20)). As required by Section 4 of that license:

- the original copyright, patent, trademark and attribution notices in these files are retained;
- files that have been modified from the upstream version (`ram/models/*_lora.py`, `scheduler/scheduling_flow_match_euler_discrete.py`, and the copied functions in `stableresfusion_utils/flux2.py`) are identified here as modified files;
- none of the upstream projects listed above ships a `NOTICE` file.

Copyright notices:

- Recognize Anything (RAM / Tag2Text): Copyright (c) 2022 OPPO.
- SeeSR / OSEDiff (DAPE): Copyright the SeeSR and OSEDiff authors.
- 🤗 Diffusers: Copyright The HuggingFace Team. All rights reserved.
- 🤗 Transformers: Copyright 2018- The Hugging Face team. All rights reserved.
- timm (PyTorch Image Models): Copyright 2019 Ross Wightman.

## 3. Non-commercial component: `my_utils/wavelet_color_fix.py`

`my_utils/wavelet_color_fix.py` is licensed under the **S-Lab License 1.0**, which permits redistribution and use **only for non-commercial purposes** (full text in [Appendix D](#appendix-d-s-lab-license-10)). The ScaleResfusion licensor cannot grant commercial rights to this file. A commercial user must either obtain permission from the S-Lab copyright holder(s) or replace this file with an implementation that is not subject to the S-Lab License.

## 4. Components that are not included in this repository

The following are downloaded separately by the user. They are not redistributed here and are governed solely by their own licenses:

- FLUX.2-klein-base-4B backbone weights (Black Forest Labs, Apache License 2.0), via Hugging Face.
- RAM checkpoint `ram_swin_large_14m.pth` ([Recognize Anything](https://github.com/xinyu1205/recognize-anything), Apache License 2.0).
- DAPE checkpoint `DAPE.pth` ([SeeSR](https://github.com/cswry/SeeSR) / [OSEDiff](https://github.com/cswry/OSEDiff), Apache License 2.0).
- Python dependencies listed in `requirements.txt` (PyTorch, diffusers, transformers, peft, accelerate, pyiqa, etc.), each under its own license.

---

## Appendix A: Apache License 2.0

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Appendix B: MIT License

### Swin Transformer

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

### ASL

```text
MIT License

Copyright (c) 2020 Alibaba-MIIL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Appendix C: BSD 3-Clause License (BLIP)

```text
Copyright (c) 2022, Salesforce.com, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of Salesforce.com nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Appendix D: S-Lab License 1.0

```text
S-Lab License 1.0

Copyright 2022 S-Lab

Redistribution and use for non-commercial purpose in source and 
binary forms, with or without modification, are permitted provided 
that the following conditions are met:

1. Redistributions of source code must retain the above copyright 
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright 
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.

3. Neither the name of the copyright holder nor the names of its 
   contributors may be used to endorse or promote products derived 
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In the event that redistribution and/or use for commercial purpose in 
source or binary forms, with or without modification is required, 
please contact the contributor(s) of the work.```
