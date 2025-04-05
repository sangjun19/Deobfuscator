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
	.ascii "1-7-1\0"
	.align	3
lC7:
	.ascii "1-8-1\0"
	.align	3
lC8:
	.ascii "2-1-1\0"
	.align	3
lC9:
	.ascii "2-2-1\0"
	.align	3
lC10:
	.ascii "2-3-1\0"
	.align	3
lC11:
	.ascii "2-4-1\0"
	.align	3
lC12:
	.ascii "2-5-1\0"
	.align	3
lC13:
	.ascii "2-6-1\0"
	.align	3
lC14:
	.ascii "2-7-1\0"
	.align	3
lC15:
	.ascii "2-8-1\0"
	.align	3
lC16:
	.ascii "3-1-1\0"
	.align	3
lC17:
	.ascii "3-2-1\0"
	.align	3
lC18:
	.ascii "3-3-1\0"
	.align	3
lC19:
	.ascii "3-4-1\0"
	.align	3
lC20:
	.ascii "3-5-1\0"
	.align	3
lC21:
	.ascii "3-6-1\0"
	.align	3
lC22:
	.ascii "3-7-1\0"
	.align	3
lC23:
	.ascii "3-8-1\0"
	.align	3
lC24:
	.ascii "4-1-1\0"
	.align	3
lC25:
	.ascii "4-2-1\0"
	.align	3
lC26:
	.ascii "4-3-1\0"
	.align	3
lC27:
	.ascii "4-4-1\0"
	.align	3
lC28:
	.ascii "4-5-1\0"
	.align	3
lC29:
	.ascii "4-6-1\0"
	.align	3
lC30:
	.ascii "4-7-1\0"
	.align	3
lC31:
	.ascii "4-8-1\0"
	.align	3
lC32:
	.ascii "5-1-1\0"
	.align	3
lC33:
	.ascii "5-2-1\0"
	.align	3
lC34:
	.ascii "5-3-1\0"
	.align	3
lC35:
	.ascii "5-4-1\0"
	.align	3
lC36:
	.ascii "5-5-1\0"
	.align	3
lC37:
	.ascii "5-6-1\0"
	.align	3
lC38:
	.ascii "5-7-1\0"
	.align	3
lC39:
	.ascii "5-8-1\0"
	.align	3
lC40:
	.ascii "6-1-1\0"
	.align	3
lC41:
	.ascii "6-2-1\0"
	.align	3
lC42:
	.ascii "6-3-1\0"
	.align	3
lC43:
	.ascii "6-4-1\0"
	.align	3
lC44:
	.ascii "6-5-1\0"
	.align	3
lC45:
	.ascii "6-6-1\0"
	.align	3
lC46:
	.ascii "6-7-1\0"
	.align	3
lC47:
	.ascii "6-8-1\0"
	.align	3
lC48:
	.ascii "7-1-1\0"
	.align	3
lC49:
	.ascii "7-2-1\0"
	.align	3
lC50:
	.ascii "7-3-1\0"
	.align	3
lC51:
	.ascii "7-4-1\0"
	.align	3
lC52:
	.ascii "7-5-1\0"
	.align	3
lC53:
	.ascii "7-6-1\0"
	.align	3
lC54:
	.ascii "7-7-1\0"
	.align	3
lC55:
	.ascii "7-8-1\0"
	.align	3
lC56:
	.ascii "8-1-1\0"
	.align	3
lC57:
	.ascii "8-2-1\0"
	.align	3
lC58:
	.ascii "8-3-1\0"
	.align	3
lC59:
	.ascii "8-4-1\0"
	.align	3
lC60:
	.ascii "8-5-1\0"
	.align	3
lC61:
	.ascii "8-6-1\0"
	.align	3
lC62:
	.ascii "8-7-1\0"
	.align	3
lC63:
	.ascii "8-8-1\0"
	.align	3
lC64:
	.ascii "9-1-1\0"
	.align	3
lC65:
	.ascii "9-2-1\0"
	.align	3
lC66:
	.ascii "9-3-1\0"
	.align	3
lC67:
	.ascii "9-4-1\0"
	.align	3
lC68:
	.ascii "9-5-1\0"
	.align	3
lC69:
	.ascii "9-6-1\0"
	.align	3
lC70:
	.ascii "9-7-1\0"
	.align	3
lC71:
	.ascii "9-8-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 9
	str	w0, [x29, 28]
	mov	w0, 8
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L11
	b	L3
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L16
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L17
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L18
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L20
	b	L166
L19:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L167
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L167
L20:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L168
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L168
L18:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L169
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L169
L17:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L170
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L170
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L171
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L171
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L172
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L172
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L173
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L173
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L174
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L174
L167:
	nop
	b	L166
L168:
	nop
	b	L166
L169:
	nop
	b	L166
L170:
	nop
	b	L166
L171:
	nop
	b	L166
L172:
	nop
	b	L166
L173:
	nop
	b	L166
L174:
	nop
	b	L166
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L175
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L175
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L175
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L175
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L175
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L175
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L37
	b	L175
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L176
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L176
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L177
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L177
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L178
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L178
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L179
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L179
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L180
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L180
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L181
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L181
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L182
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L182
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L183
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L183
L176:
	nop
	b	L175
L177:
	nop
	b	L175
L178:
	nop
	b	L175
L179:
	nop
	b	L175
L180:
	nop
	b	L175
L181:
	nop
	b	L175
L182:
	nop
	b	L175
L183:
	nop
	b	L175
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L46
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L48
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L50
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L54
	b	L184
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L185
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L185
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L186
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L186
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L187
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L187
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L188
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L188
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L189
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L189
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L190
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L190
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L191
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L191
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L192
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L192
L185:
	nop
	b	L184
L186:
	nop
	b	L184
L187:
	nop
	b	L184
L188:
	nop
	b	L184
L189:
	nop
	b	L184
L190:
	nop
	b	L184
L191:
	nop
	b	L184
L192:
	nop
	b	L184
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L63
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L193
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L71
	b	L193
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L194
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L194
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L195
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L195
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L196
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L196
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L197
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L197
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L198
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L198
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L199
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L199
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L200
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	nop
	b	L200
L63:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L201
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
	b	L201
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
L200:
	nop
	b	L193
L201:
	nop
	b	L193
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L80
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L202
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L82
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L202
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L83
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L202
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L84
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L202
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L85
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L202
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L86
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L202
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L88
	b	L202
L87:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L203
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
	b	L203
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L204
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
	b	L204
L86:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L205
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
	b	L205
L85:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L206
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
	b	L206
L84:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L207
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	nop
	b	L207
L83:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L208
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
	b	L208
L82:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L209
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
	b	L209
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L210
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
	b	L210
L203:
	nop
	b	L202
L204:
	nop
	b	L202
L205:
	nop
	b	L202
L206:
	nop
	b	L202
L207:
	nop
	b	L202
L208:
	nop
	b	L202
L209:
	nop
	b	L202
L210:
	nop
	b	L202
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L97
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L211
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L99
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L211
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L100
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L211
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L101
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L211
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L102
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L211
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L103
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L211
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L105
	b	L211
L104:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L212
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	nop
	b	L212
L105:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L213
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
	b	L213
L103:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L214
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	nop
	b	L214
L102:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L215
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
	b	L215
L101:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L216
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
	b	L216
L100:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L217
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
	b	L217
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L218
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	nop
	b	L218
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L219
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
	b	L219
L212:
	nop
	b	L211
L213:
	nop
	b	L211
L214:
	nop
	b	L211
L215:
	nop
	b	L211
L216:
	nop
	b	L211
L217:
	nop
	b	L211
L218:
	nop
	b	L211
L219:
	nop
	b	L211
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L114
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L220
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L116
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L220
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L117
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L220
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L118
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L220
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L119
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L220
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L120
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L220
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L122
	b	L220
L121:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L221
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	nop
	b	L221
L122:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L222
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
	b	L222
L120:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L223
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	nop
	b	L223
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L224
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
	b	L224
L118:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L225
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	nop
	b	L225
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L226
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
	b	L226
L116:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L227
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	nop
	b	L227
L114:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L228
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
	b	L228
L221:
	nop
	b	L220
L222:
	nop
	b	L220
L223:
	nop
	b	L220
L224:
	nop
	b	L220
L225:
	nop
	b	L220
L226:
	nop
	b	L220
L227:
	nop
	b	L220
L228:
	nop
	b	L220
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L131
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L229
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L133
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L229
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L134
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L229
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L135
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L229
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L136
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L229
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L137
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L229
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L138
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L139
	b	L229
L138:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L230
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	nop
	b	L230
L139:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L231
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
	b	L231
L137:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L232
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	nop
	b	L232
L136:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L233
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
	b	L233
L135:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L234
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	nop
	b	L234
L134:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L235
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
	b	L235
L133:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L236
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	nop
	b	L236
L131:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L237
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
	b	L237
L230:
	nop
	b	L229
L231:
	nop
	b	L229
L232:
	nop
	b	L229
L233:
	nop
	b	L229
L234:
	nop
	b	L229
L235:
	nop
	b	L229
L236:
	nop
	b	L229
L237:
	nop
	b	L229
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L148
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L238
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L150
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L238
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L151
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L238
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L152
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L238
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L153
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L238
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L154
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L238
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L155
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L156
	b	L238
L155:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L239
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	nop
	b	L239
L156:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L240
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	nop
	b	L240
L154:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L241
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	nop
	b	L241
L153:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L242
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
	b	L242
L152:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L243
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	nop
	b	L243
L151:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L244
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	nop
	b	L244
L150:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L245
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	nop
	b	L245
L148:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L246
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
	b	L246
L239:
	nop
	b	L238
L240:
	nop
	b	L238
L241:
	nop
	b	L238
L242:
	nop
	b	L238
L243:
	nop
	b	L238
L244:
	nop
	b	L238
L245:
	nop
	b	L238
L246:
	nop
	b	L238
L166:
	nop
	b	L3
L175:
	nop
	b	L3
L184:
	nop
	b	L3
L193:
	nop
	b	L3
L202:
	nop
	b	L3
L211:
	nop
	b	L3
L220:
	nop
	b	L3
L229:
	nop
	b	L3
L238:
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
