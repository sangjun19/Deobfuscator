	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-1-2\0"
	.align	3
lC2:
	.ascii "1-1-3\0"
	.align	3
lC3:
	.ascii "1-1-4\0"
	.align	3
lC4:
	.ascii "1-1-5\0"
	.align	3
lC5:
	.ascii "1-1-6\0"
	.align	3
lC6:
	.ascii "1-1-7\0"
	.align	3
lC7:
	.ascii "1-1-8\0"
	.align	3
lC8:
	.ascii "1-1-9\0"
	.align	3
lC9:
	.ascii "2-1-1\0"
	.align	3
lC10:
	.ascii "2-1-2\0"
	.align	3
lC11:
	.ascii "2-1-3\0"
	.align	3
lC12:
	.ascii "2-1-4\0"
	.align	3
lC13:
	.ascii "2-1-5\0"
	.align	3
lC14:
	.ascii "2-1-6\0"
	.align	3
lC15:
	.ascii "2-1-7\0"
	.align	3
lC16:
	.ascii "2-1-8\0"
	.align	3
lC17:
	.ascii "2-1-9\0"
	.align	3
lC18:
	.ascii "3-1-1\0"
	.align	3
lC19:
	.ascii "3-1-2\0"
	.align	3
lC20:
	.ascii "3-1-3\0"
	.align	3
lC21:
	.ascii "3-1-4\0"
	.align	3
lC22:
	.ascii "3-1-5\0"
	.align	3
lC23:
	.ascii "3-1-6\0"
	.align	3
lC24:
	.ascii "3-1-7\0"
	.align	3
lC25:
	.ascii "3-1-8\0"
	.align	3
lC26:
	.ascii "3-1-9\0"
	.align	3
lC27:
	.ascii "4-1-1\0"
	.align	3
lC28:
	.ascii "4-1-2\0"
	.align	3
lC29:
	.ascii "4-1-3\0"
	.align	3
lC30:
	.ascii "4-1-4\0"
	.align	3
lC31:
	.ascii "4-1-5\0"
	.align	3
lC32:
	.ascii "4-1-6\0"
	.align	3
lC33:
	.ascii "4-1-7\0"
	.align	3
lC34:
	.ascii "4-1-8\0"
	.align	3
lC35:
	.ascii "4-1-9\0"
	.align	3
lC36:
	.ascii "5-1-1\0"
	.align	3
lC37:
	.ascii "5-1-2\0"
	.align	3
lC38:
	.ascii "5-1-3\0"
	.align	3
lC39:
	.ascii "5-1-4\0"
	.align	3
lC40:
	.ascii "5-1-5\0"
	.align	3
lC41:
	.ascii "5-1-6\0"
	.align	3
lC42:
	.ascii "5-1-7\0"
	.align	3
lC43:
	.ascii "5-1-8\0"
	.align	3
lC44:
	.ascii "5-1-9\0"
	.align	3
lC45:
	.ascii "6-1-1\0"
	.align	3
lC46:
	.ascii "6-1-2\0"
	.align	3
lC47:
	.ascii "6-1-3\0"
	.align	3
lC48:
	.ascii "6-1-4\0"
	.align	3
lC49:
	.ascii "6-1-5\0"
	.align	3
lC50:
	.ascii "6-1-6\0"
	.align	3
lC51:
	.ascii "6-1-7\0"
	.align	3
lC52:
	.ascii "6-1-8\0"
	.align	3
lC53:
	.ascii "6-1-9\0"
	.align	3
lC54:
	.ascii "7-1-1\0"
	.align	3
lC55:
	.ascii "7-1-2\0"
	.align	3
lC56:
	.ascii "7-1-3\0"
	.align	3
lC57:
	.ascii "7-1-4\0"
	.align	3
lC58:
	.ascii "7-1-5\0"
	.align	3
lC59:
	.ascii "7-1-6\0"
	.align	3
lC60:
	.ascii "7-1-7\0"
	.align	3
lC61:
	.ascii "7-1-8\0"
	.align	3
lC62:
	.ascii "7-1-9\0"
	.align	3
lC63:
	.ascii "8-1-1\0"
	.align	3
lC64:
	.ascii "8-1-2\0"
	.align	3
lC65:
	.ascii "8-1-3\0"
	.align	3
lC66:
	.ascii "8-1-4\0"
	.align	3
lC67:
	.ascii "8-1-5\0"
	.align	3
lC68:
	.ascii "8-1-6\0"
	.align	3
lC69:
	.ascii "8-1-7\0"
	.align	3
lC70:
	.ascii "8-1-8\0"
	.align	3
lC71:
	.ascii "8-1-9\0"
	.align	3
lC72:
	.ascii "9-1-1\0"
	.align	3
lC73:
	.ascii "9-1-2\0"
	.align	3
lC74:
	.ascii "9-1-3\0"
	.align	3
lC75:
	.ascii "9-1-4\0"
	.align	3
lC76:
	.ascii "9-1-5\0"
	.align	3
lC77:
	.ascii "9-1-6\0"
	.align	3
lC78:
	.ascii "9-1-7\0"
	.align	3
lC79:
	.ascii "9-1-8\0"
	.align	3
lC80:
	.ascii "9-1-9\0"
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
	mov	w0, 1
	str	w0, [x29, 24]
	mov	w0, 9
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
	cmp	w0, 1
	bne	L112
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L113
L21:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L14
L22:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L14
L20:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L14
L19:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L14
L18:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L14
L17:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L14
L16:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L14
L15:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L14
L13:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L14:
L113:
	nop
	b	L112
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L114
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L115
L32:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L25
L33:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L25
L31:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L25
L30:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L25
L29:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L25
L28:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L25
L27:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L25
L26:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L25:
L115:
	nop
	b	L114
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L116
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L117
L43:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L36
L44:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L36
L42:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L36
L41:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L36
L40:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L36
L39:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	b	L36
L38:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L36
L37:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L36
L35:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L36:
L117:
	nop
	b	L116
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L118
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L52
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L55
	b	L119
L54:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L47
L55:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L47
L53:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L47
L52:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L47
L51:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L47
L50:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L47
L49:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L47
L48:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L47:
L119:
	nop
	b	L118
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L120
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L121
L65:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L58
L66:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L58
L64:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L58
L63:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	b	L58
L62:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L58
L61:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L58
L60:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L58
L59:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L58
L57:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L58:
L121:
	nop
	b	L120
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L122
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L72
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L77
	b	L123
L76:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L69
L77:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L69
L75:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	b	L69
L74:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L69
L73:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L69
L72:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L69
L71:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	b	L69
L70:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L69
L68:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L69:
L123:
	nop
	b	L122
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L124
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L88
	b	L125
L87:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L80
L88:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	b	L80
L86:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L80
L85:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L80
L84:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L80
L83:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	b	L80
L82:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L80
L81:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	b	L80
L79:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	nop
L80:
L125:
	nop
	b	L124
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L126
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L92
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L94
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L96
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L98
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L99
	b	L127
L98:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	b	L91
L99:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L91
L97:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	b	L91
L96:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L91
L95:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	b	L91
L94:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L91
L93:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	b	L91
L92:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L91
L90:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L91:
L127:
	nop
	b	L126
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L128
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L106
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L110
	b	L129
L109:
	adrp	x0, lC72@PAGE
	add	x0, x0, lC72@PAGEOFF;
	bl	_puts
	b	L102
L110:
	adrp	x0, lC73@PAGE
	add	x0, x0, lC73@PAGEOFF;
	bl	_puts
	b	L102
L108:
	adrp	x0, lC74@PAGE
	add	x0, x0, lC74@PAGEOFF;
	bl	_puts
	b	L102
L107:
	adrp	x0, lC75@PAGE
	add	x0, x0, lC75@PAGEOFF;
	bl	_puts
	b	L102
L106:
	adrp	x0, lC76@PAGE
	add	x0, x0, lC76@PAGEOFF;
	bl	_puts
	b	L102
L105:
	adrp	x0, lC77@PAGE
	add	x0, x0, lC77@PAGEOFF;
	bl	_puts
	b	L102
L104:
	adrp	x0, lC78@PAGE
	add	x0, x0, lC78@PAGEOFF;
	bl	_puts
	b	L102
L103:
	adrp	x0, lC79@PAGE
	add	x0, x0, lC79@PAGEOFF;
	bl	_puts
	b	L102
L101:
	adrp	x0, lC80@PAGE
	add	x0, x0, lC80@PAGEOFF;
	bl	_puts
	nop
L102:
L129:
	nop
	b	L128
L112:
	nop
	b	L3
L114:
	nop
	b	L3
L116:
	nop
	b	L3
L118:
	nop
	b	L3
L120:
	nop
	b	L3
L122:
	nop
	b	L3
L124:
	nop
	b	L3
L126:
	nop
	b	L3
L128:
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
