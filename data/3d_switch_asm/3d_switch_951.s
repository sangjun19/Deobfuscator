	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-2-1\0"
	.align	3
lC2:
	.ascii "1-3-1\0"
	.align	3
lC3:
	.ascii "1-4-1\0"
	.align	3
lC4:
	.ascii "1-5-1\0"
	.align	3
lC5:
	.ascii "1-6-1\0"
	.align	3
lC6:
	.ascii "2-1-1\0"
	.align	3
lC7:
	.ascii "2-2-1\0"
	.align	3
lC8:
	.ascii "2-3-1\0"
	.align	3
lC9:
	.ascii "2-4-1\0"
	.align	3
lC10:
	.ascii "2-5-1\0"
	.align	3
lC11:
	.ascii "2-6-1\0"
	.align	3
lC12:
	.ascii "3-1-1\0"
	.align	3
lC13:
	.ascii "3-2-1\0"
	.align	3
lC14:
	.ascii "3-3-1\0"
	.align	3
lC15:
	.ascii "3-4-1\0"
	.align	3
lC16:
	.ascii "3-5-1\0"
	.align	3
lC17:
	.ascii "3-6-1\0"
	.align	3
lC18:
	.ascii "4-1-1\0"
	.align	3
lC19:
	.ascii "4-2-1\0"
	.align	3
lC20:
	.ascii "4-3-1\0"
	.align	3
lC21:
	.ascii "4-4-1\0"
	.align	3
lC22:
	.ascii "4-5-1\0"
	.align	3
lC23:
	.ascii "4-6-1\0"
	.align	3
lC24:
	.ascii "5-1-1\0"
	.align	3
lC25:
	.ascii "5-2-1\0"
	.align	3
lC26:
	.ascii "5-3-1\0"
	.align	3
lC27:
	.ascii "5-4-1\0"
	.align	3
lC28:
	.ascii "5-5-1\0"
	.align	3
lC29:
	.ascii "5-6-1\0"
	.align	3
lC30:
	.ascii "6-1-1\0"
	.align	3
lC31:
	.ascii "6-2-1\0"
	.align	3
lC32:
	.ascii "6-3-1\0"
	.align	3
lC33:
	.ascii "6-4-1\0"
	.align	3
lC34:
	.ascii "6-5-1\0"
	.align	3
lC35:
	.ascii "6-6-1\0"
	.align	3
lC36:
	.ascii "7-1-1\0"
	.align	3
lC37:
	.ascii "7-2-1\0"
	.align	3
lC38:
	.ascii "7-3-1\0"
	.align	3
lC39:
	.ascii "7-4-1\0"
	.align	3
lC40:
	.ascii "7-5-1\0"
	.align	3
lC41:
	.ascii "7-6-1\0"
	.align	3
lC42:
	.ascii "8-1-1\0"
	.align	3
lC43:
	.ascii "8-2-1\0"
	.align	3
lC44:
	.ascii "8-3-1\0"
	.align	3
lC45:
	.ascii "8-4-1\0"
	.align	3
lC46:
	.ascii "8-5-1\0"
	.align	3
lC47:
	.ascii "8-6-1\0"
	.align	3
lC48:
	.ascii "9-1-1\0"
	.align	3
lC49:
	.ascii "9-2-1\0"
	.align	3
lC50:
	.ascii "9-3-1\0"
	.align	3
lC51:
	.ascii "9-4-1\0"
	.align	3
lC52:
	.ascii "9-5-1\0"
	.align	3
lC53:
	.ascii "9-6-1\0"
	.align	3
lC54:
	.ascii "10-1-1\0"
	.align	3
lC55:
	.ascii "10-2-1\0"
	.align	3
lC56:
	.ascii "10-3-1\0"
	.align	3
lC57:
	.ascii "10-4-1\0"
	.align	3
lC58:
	.ascii "10-5-1\0"
	.align	3
lC59:
	.ascii "10-6-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 10
	str	w0, [x29, 28]
	mov	w0, 6
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 10
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 10
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L12
	b	L3
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L16
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L17
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L19
	b	L144
L18:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L145
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L145
L19:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L146
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L146
L17:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L147
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L147
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L148
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L148
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L149
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L149
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L150
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L150
L145:
	nop
	b	L144
L146:
	nop
	b	L144
L147:
	nop
	b	L144
L148:
	nop
	b	L144
L149:
	nop
	b	L144
L150:
	nop
	b	L144
L12:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L151
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L151
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L151
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L151
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L32
	b	L151
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L152
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L152
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L153
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L153
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L154
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L154
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L155
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L155
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L156
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L156
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L157
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L157
L152:
	nop
	b	L151
L153:
	nop
	b	L151
L154:
	nop
	b	L151
L155:
	nop
	b	L151
L156:
	nop
	b	L151
L157:
	nop
	b	L151
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L42
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L45
	b	L158
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L159
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L159
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L160
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L160
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L161
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L161
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L162
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L162
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L163
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L163
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L164
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L164
L159:
	nop
	b	L158
L160:
	nop
	b	L158
L161:
	nop
	b	L158
L162:
	nop
	b	L158
L163:
	nop
	b	L158
L164:
	nop
	b	L158
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L165
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L165
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L165
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L56
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L165
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L58
	b	L165
L57:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L166
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L166
L58:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L167
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L167
L56:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L168
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L168
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L169
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L169
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L170
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L170
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L171
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L171
L166:
	nop
	b	L165
L167:
	nop
	b	L165
L168:
	nop
	b	L165
L169:
	nop
	b	L165
L170:
	nop
	b	L165
L171:
	nop
	b	L165
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L172
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L172
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L172
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L172
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L71
	b	L172
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L173
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L173
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L174
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L174
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L175
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L175
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L176
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L176
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L177
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L177
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L178
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L178
L173:
	nop
	b	L172
L174:
	nop
	b	L172
L175:
	nop
	b	L172
L176:
	nop
	b	L172
L177:
	nop
	b	L172
L178:
	nop
	b	L172
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L179
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L80
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L179
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L81
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L179
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L82
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L179
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L84
	b	L179
L83:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L180
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	nop
	b	L180
L84:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L181
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
	b	L181
L82:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L182
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
	b	L182
L81:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L183
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
	b	L183
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L184
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
	b	L184
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L185
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
	b	L185
L180:
	nop
	b	L179
L181:
	nop
	b	L179
L182:
	nop
	b	L179
L183:
	nop
	b	L179
L184:
	nop
	b	L179
L185:
	nop
	b	L179
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L93
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L94
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L95
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L97
	b	L186
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L187
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	nop
	b	L187
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L188
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
	b	L188
L95:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L189
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
	b	L189
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L190
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
	b	L190
L93:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L191
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	nop
	b	L191
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L192
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
	b	L192
L187:
	nop
	b	L186
L188:
	nop
	b	L186
L189:
	nop
	b	L186
L190:
	nop
	b	L186
L191:
	nop
	b	L186
L192:
	nop
	b	L186
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L104
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L106
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L107
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L108
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L110
	b	L193
L109:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L194
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	nop
	b	L194
L110:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L195
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
	b	L195
L108:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L196
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
	b	L196
L107:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L197
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
	b	L197
L106:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L198
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	nop
	b	L198
L104:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L199
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
	b	L199
L194:
	nop
	b	L193
L195:
	nop
	b	L193
L196:
	nop
	b	L193
L197:
	nop
	b	L193
L198:
	nop
	b	L193
L199:
	nop
	b	L193
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L117
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L200
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L119
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L200
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L120
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L200
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L121
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L200
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L122
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L123
	b	L200
L122:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L201
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	nop
	b	L201
L123:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L202
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
	b	L202
L121:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L203
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	nop
	b	L203
L120:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L204
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
	b	L204
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L205
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	nop
	b	L205
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L206
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
	b	L206
L201:
	nop
	b	L200
L202:
	nop
	b	L200
L203:
	nop
	b	L200
L204:
	nop
	b	L200
L205:
	nop
	b	L200
L206:
	nop
	b	L200
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L130
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L207
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L132
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L207
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L133
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L207
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L134
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L207
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L135
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L136
	b	L207
L135:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L208
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	nop
	b	L208
L136:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L209
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
	b	L209
L134:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L210
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	nop
	b	L210
L133:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L211
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
	b	L211
L132:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L212
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	nop
	b	L212
L130:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L213
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
	b	L213
L208:
	nop
	b	L207
L209:
	nop
	b	L207
L210:
	nop
	b	L207
L211:
	nop
	b	L207
L212:
	nop
	b	L207
L213:
	nop
	b	L207
L144:
	nop
	b	L3
L151:
	nop
	b	L3
L158:
	nop
	b	L3
L165:
	nop
	b	L3
L172:
	nop
	b	L3
L179:
	nop
	b	L3
L186:
	nop
	b	L3
L193:
	nop
	b	L3
L200:
	nop
	b	L3
L207:
	nop
L3:
	mov	w0, 0
	ldp	x29, x30, [sp], 32
LCFI2:
	ret
LFE1:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.uleb128 0
	.byte	0x4
	.set L$set$3,LCFI0-LFB1
	.long L$set$3
	.byte	0xe
	.uleb128 0x20
	.byte	0x9d
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x3
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE1:
	.ident	"GCC: (Homebrew GCC 14.2.0_1) 14.2.0"
	.subsections_via_symbols
