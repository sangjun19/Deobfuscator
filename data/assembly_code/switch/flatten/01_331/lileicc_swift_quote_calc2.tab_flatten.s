	.file	"lileicc_swift_quote_calc2.tab_flatten.c"
	.text
	.local	quote_calc2_sindex
	.comm	quote_calc2_sindex,66,32
	.local	quote_calc2_gindex
	.comm	quote_calc2_gindex,8,8
	.local	quote_calc2_dgoto
	.comm	quote_calc2_dgoto,8,8
	.globl	quote_calc2_val
	.bss
	.align 4
	.type	quote_calc2_val, @object
	.size	quote_calc2_val, 4
quote_calc2_val:
	.zero	4
	.globl	_TIG_IZ_EK2N_argv
	.align 8
	.type	_TIG_IZ_EK2N_argv, @object
	.size	_TIG_IZ_EK2N_argv, 8
_TIG_IZ_EK2N_argv:
	.zero	8
	.local	quote_calc2_defred
	.comm	quote_calc2_defred,66,32
	.local	yysccsid
	.comm	yysccsid,36,32
	.local	quote_calc2_len
	.comm	quote_calc2_len,38,32
	.globl	regs
	.align 32
	.type	regs, @object
	.size	regs, 104
regs:
	.zero	104
	.globl	quote_calc2_debug
	.align 4
	.type	quote_calc2_debug, @object
	.size	quote_calc2_debug, 4
quote_calc2_debug:
	.zero	4
	.local	quote_calc2_table
	.comm	quote_calc2_table,520,32
	.globl	quote_calc2_char
	.align 4
	.type	quote_calc2_char, @object
	.size	quote_calc2_char, 4
quote_calc2_char:
	.zero	4
	.globl	base
	.align 4
	.type	base, @object
	.size	base, 4
base:
	.zero	4
	.globl	quote_calc2_lval
	.align 4
	.type	quote_calc2_lval, @object
	.size	quote_calc2_lval, 4
quote_calc2_lval:
	.zero	4
	.local	quote_calc2_check
	.comm	quote_calc2_check,520,32
	.globl	_TIG_IZ_EK2N_envp
	.align 8
	.type	_TIG_IZ_EK2N_envp, @object
	.size	_TIG_IZ_EK2N_envp, 8
_TIG_IZ_EK2N_envp:
	.zero	8
	.local	yystack
	.comm	yystack,48,32
	.globl	quote_calc2_errflag
	.align 4
	.type	quote_calc2_errflag, @object
	.size	quote_calc2_errflag, 4
quote_calc2_errflag:
	.zero	4
	.globl	_TIG_IZ_EK2N_argc
	.align 4
	.type	_TIG_IZ_EK2N_argc, @object
	.size	_TIG_IZ_EK2N_argc, 4
_TIG_IZ_EK2N_argc:
	.zero	4
	.globl	quote_calc2_nerrs
	.align 4
	.type	quote_calc2_nerrs, @object
	.size	quote_calc2_nerrs, 4
quote_calc2_nerrs:
	.zero	4
	.local	quote_calc2_lhs
	.comm	quote_calc2_lhs,38,32
	.local	quote_calc2_rindex
	.comm	quote_calc2_rindex,66,32
	.text
	.globl	quote_calc2_lex
	.type	quote_calc2_lex, @function
quote_calc2_lex:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$12, -8(%rbp)
.L24:
	cmpq	$12, -8(%rbp)
	ja	.L25
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L25-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L12:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L16
.L3:
	movq	$8, -8(%rbp)
	jmp	.L16
.L8:
	call	getchar@PLT
	movl	%eax, -28(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L16
.L14:
	movl	-28(%rbp), %eax
	jmp	.L17
.L5:
	movl	$270, %eax
	jmp	.L17
.L7:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L18
	movq	$6, -8(%rbp)
	jmp	.L16
.L18:
	movq	$1, -8(%rbp)
	jmp	.L16
.L10:
	movl	-28(%rbp), %eax
	subl	$48, %eax
	movl	%eax, quote_calc2_lval(%rip)
	movq	$10, -8(%rbp)
	jmp	.L16
.L11:
	cmpl	$32, -28(%rbp)
	jne	.L20
	movq	$8, -8(%rbp)
	jmp	.L16
.L20:
	movq	$0, -8(%rbp)
	jmp	.L16
.L6:
	movl	$269, %eax
	jmp	.L17
.L15:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L16
.L9:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$512, %eax
	testl	%eax, %eax
	je	.L22
	movq	$2, -8(%rbp)
	jmp	.L16
.L22:
	movq	$4, -8(%rbp)
	jmp	.L16
.L13:
	movl	-28(%rbp), %eax
	subl	$97, %eax
	movl	%eax, quote_calc2_lval(%rip)
	movq	$11, -8(%rbp)
	jmp	.L16
.L25:
	nop
.L16:
	jmp	.L24
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	quote_calc2_lex, .-quote_calc2_lex
	.type	yygrowstack, @function
yygrowstack:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$13, -24(%rbp)
.L57:
	cmpq	$18, -24(%rbp)
	ja	.L58
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L44-.L29
	.long	.L58-.L29
	.long	.L43-.L29
	.long	.L42-.L29
	.long	.L41-.L29
	.long	.L58-.L29
	.long	.L40-.L29
	.long	.L58-.L29
	.long	.L39-.L29
	.long	.L38-.L29
	.long	.L37-.L29
	.long	.L36-.L29
	.long	.L35-.L29
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L28-.L29
	.text
.L28:
	cmpl	$0, -44(%rbp)
	jne	.L45
	movq	$2, -24(%rbp)
	jmp	.L47
.L45:
	movq	$15, -24(%rbp)
	jmp	.L47
.L41:
	movl	$0, %eax
	jmp	.L48
.L33:
	cmpl	$10000, -44(%rbp)
	jbe	.L49
	movq	$6, -24(%rbp)
	jmp	.L47
.L49:
	movq	$12, -24(%rbp)
	jmp	.L47
.L32:
	cmpl	$9999, -44(%rbp)
	jbe	.L51
	movq	$8, -24(%rbp)
	jmp	.L47
.L51:
	movq	$9, -24(%rbp)
	jmp	.L47
.L35:
	movq	-56(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rcx
	movq	%rdx, %rax
	subq	%rcx, %rax
	sarq	%rax
	movl	%eax, -48(%rbp)
	movl	-44(%rbp), %eax
	leaq	(%rax,%rax), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$17, -24(%rbp)
	jmp	.L47
.L39:
	movl	$-1, %eax
	jmp	.L48
.L42:
	movq	-56(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 32(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 40(%rax)
	movq	-56(%rbp), %rax
	movl	-44(%rbp), %edx
	movl	%edx, (%rax)
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-44(%rbp), %eax
	addq	%rax, %rax
	subq	$2, %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 24(%rax)
	movq	$4, -24(%rbp)
	jmp	.L47
.L31:
	cmpq	$0, -32(%rbp)
	jne	.L53
	movq	$10, -24(%rbp)
	jmp	.L47
.L53:
	movq	$3, -24(%rbp)
	jmp	.L47
.L36:
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 16(%rax)
	movl	-44(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	movq	32(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$16, -24(%rbp)
	jmp	.L47
.L38:
	sall	-44(%rbp)
	movq	$14, -24(%rbp)
	jmp	.L47
.L34:
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -44(%rbp)
	movq	$18, -24(%rbp)
	jmp	.L47
.L30:
	cmpq	$0, -40(%rbp)
	jne	.L55
	movq	$0, -24(%rbp)
	jmp	.L47
.L55:
	movq	$11, -24(%rbp)
	jmp	.L47
.L40:
	movl	$10000, -44(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L47
.L37:
	movl	$-1, %eax
	jmp	.L48
.L44:
	movl	$-1, %eax
	jmp	.L48
.L43:
	movl	$200, -44(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L47
.L58:
	nop
.L47:
	jmp	.L57
.L48:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	yygrowstack, .-yygrowstack
	.section	.rodata
.LC0:
	.string	"%s\n"
	.text
	.type	quote_calc2_error, @function
quote_calc2_error:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L64:
	cmpq	$0, -8(%rbp)
	je	.L60
	cmpq	$1, -8(%rbp)
	jne	.L66
	jmp	.L65
.L60:
	movq	stderr(%rip), %rax
	movq	-24(%rbp), %rdx
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$1, -8(%rbp)
	jmp	.L63
.L66:
	nop
.L63:
	jmp	.L64
.L65:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	quote_calc2_error, .-quote_calc2_error
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, yystack(%rip)
	movq	$0, 8+yystack(%rip)
	movq	$0, 16+yystack(%rip)
	movq	$0, 24+yystack(%rip)
	movq	$0, 32+yystack(%rip)
	movq	$0, 40+yystack(%rip)
	nop
.L68:
	movl	$0, quote_calc2_lval(%rip)
	nop
.L69:
	movl	$0, quote_calc2_val(%rip)
	nop
.L70:
	movl	$0, quote_calc2_char(%rip)
	nop
.L71:
	movl	$0, quote_calc2_errflag(%rip)
	nop
.L72:
	movl	$0, quote_calc2_nerrs(%rip)
	nop
.L73:
	movl	$0, quote_calc2_debug(%rip)
	nop
.L74:
	movw	$10, quote_calc2_check(%rip)
	movw	$10, 2+quote_calc2_check(%rip)
	movw	$40, 4+quote_calc2_check(%rip)
	movw	$124, 6+quote_calc2_check(%rip)
	movw	$40, 8+quote_calc2_check(%rip)
	movw	$10, 10+quote_calc2_check(%rip)
	movw	$10, 12+quote_calc2_check(%rip)
	movw	$10, 14+quote_calc2_check(%rip)
	movw	$10, 16+quote_calc2_check(%rip)
	movw	$10, 18+quote_calc2_check(%rip)
	movw	$61, 20+quote_calc2_check(%rip)
	movw	$10, 22+quote_calc2_check(%rip)
	movw	$10, 24+quote_calc2_check(%rip)
	movw	$10, 26+quote_calc2_check(%rip)
	movw	$10, 28+quote_calc2_check(%rip)
	movw	$258, 30+quote_calc2_check(%rip)
	movw	$10, 32+quote_calc2_check(%rip)
	movw	$260, 34+quote_calc2_check(%rip)
	movw	$41, 36+quote_calc2_check(%rip)
	movw	$262, 38+quote_calc2_check(%rip)
	movw	$269, 40+quote_calc2_check(%rip)
	movw	$264, 42+quote_calc2_check(%rip)
	movw	$10, 44+quote_calc2_check(%rip)
	movw	$266, 46+quote_calc2_check(%rip)
	movw	$10, 48+quote_calc2_check(%rip)
	movw	$268, 50+quote_calc2_check(%rip)
	movw	$-1, 52+quote_calc2_check(%rip)
	movw	$-1, 54+quote_calc2_check(%rip)
	movw	$-1, 56+quote_calc2_check(%rip)
	movw	$-1, 58+quote_calc2_check(%rip)
	movw	$-1, 60+quote_calc2_check(%rip)
	movw	$41, 62+quote_calc2_check(%rip)
	movw	$-1, 64+quote_calc2_check(%rip)
	movw	$-1, 66+quote_calc2_check(%rip)
	movw	$-1, 68+quote_calc2_check(%rip)
	movw	$-1, 70+quote_calc2_check(%rip)
	movw	$41, 72+quote_calc2_check(%rip)
	movw	$41, 74+quote_calc2_check(%rip)
	movw	$41, 76+quote_calc2_check(%rip)
	movw	$41, 78+quote_calc2_check(%rip)
	movw	$41, 80+quote_calc2_check(%rip)
	movw	$-1, 82+quote_calc2_check(%rip)
	movw	$41, 84+quote_calc2_check(%rip)
	movw	$41, 86+quote_calc2_check(%rip)
	movw	$41, 88+quote_calc2_check(%rip)
	movw	$3, 90+quote_calc2_check(%rip)
	movw	$-1, 92+quote_calc2_check(%rip)
	movw	$-1, 94+quote_calc2_check(%rip)
	movw	$6, 96+quote_calc2_check(%rip)
	movw	$-1, 98+quote_calc2_check(%rip)
	movw	$-1, 100+quote_calc2_check(%rip)
	movw	$-1, 102+quote_calc2_check(%rip)
	movw	$-1, 104+quote_calc2_check(%rip)
	movw	$-1, 106+quote_calc2_check(%rip)
	movw	$-1, 108+quote_calc2_check(%rip)
	movw	$13, 110+quote_calc2_check(%rip)
	movw	$-1, 112+quote_calc2_check(%rip)
	movw	$-1, 114+quote_calc2_check(%rip)
	movw	$16, 116+quote_calc2_check(%rip)
	movw	$17, 118+quote_calc2_check(%rip)
	movw	$18, 120+quote_calc2_check(%rip)
	movw	$19, 122+quote_calc2_check(%rip)
	movw	$20, 124+quote_calc2_check(%rip)
	movw	$21, 126+quote_calc2_check(%rip)
	movw	$22, 128+quote_calc2_check(%rip)
	movw	$-1, 130+quote_calc2_check(%rip)
	movw	$-1, 132+quote_calc2_check(%rip)
	movw	$-1, 134+quote_calc2_check(%rip)
	movw	$-1, 136+quote_calc2_check(%rip)
	movw	$-1, 138+quote_calc2_check(%rip)
	movw	$-1, 140+quote_calc2_check(%rip)
	movw	$-1, 142+quote_calc2_check(%rip)
	movw	$-1, 144+quote_calc2_check(%rip)
	movw	$-1, 146+quote_calc2_check(%rip)
	movw	$-1, 148+quote_calc2_check(%rip)
	movw	$-1, 150+quote_calc2_check(%rip)
	movw	$-1, 152+quote_calc2_check(%rip)
	movw	$-1, 154+quote_calc2_check(%rip)
	movw	$-1, 156+quote_calc2_check(%rip)
	movw	$-1, 158+quote_calc2_check(%rip)
	movw	$-1, 160+quote_calc2_check(%rip)
	movw	$-1, 162+quote_calc2_check(%rip)
	movw	$-1, 164+quote_calc2_check(%rip)
	movw	$-1, 166+quote_calc2_check(%rip)
	movw	$-1, 168+quote_calc2_check(%rip)
	movw	$-1, 170+quote_calc2_check(%rip)
	movw	$-1, 172+quote_calc2_check(%rip)
	movw	$-1, 174+quote_calc2_check(%rip)
	movw	$-1, 176+quote_calc2_check(%rip)
	movw	$-1, 178+quote_calc2_check(%rip)
	movw	$-1, 180+quote_calc2_check(%rip)
	movw	$-1, 182+quote_calc2_check(%rip)
	movw	$-1, 184+quote_calc2_check(%rip)
	movw	$-1, 186+quote_calc2_check(%rip)
	movw	$-1, 188+quote_calc2_check(%rip)
	movw	$-1, 190+quote_calc2_check(%rip)
	movw	$-1, 192+quote_calc2_check(%rip)
	movw	$-1, 194+quote_calc2_check(%rip)
	movw	$-1, 196+quote_calc2_check(%rip)
	movw	$-1, 198+quote_calc2_check(%rip)
	movw	$-1, 200+quote_calc2_check(%rip)
	movw	$124, 202+quote_calc2_check(%rip)
	movw	$-1, 204+quote_calc2_check(%rip)
	movw	$-1, 206+quote_calc2_check(%rip)
	movw	$-1, 208+quote_calc2_check(%rip)
	movw	$-1, 210+quote_calc2_check(%rip)
	movw	$-1, 212+quote_calc2_check(%rip)
	movw	$-1, 214+quote_calc2_check(%rip)
	movw	$-1, 216+quote_calc2_check(%rip)
	movw	$-1, 218+quote_calc2_check(%rip)
	movw	$-1, 220+quote_calc2_check(%rip)
	movw	$-1, 222+quote_calc2_check(%rip)
	movw	$-1, 224+quote_calc2_check(%rip)
	movw	$-1, 226+quote_calc2_check(%rip)
	movw	$124, 228+quote_calc2_check(%rip)
	movw	$124, 230+quote_calc2_check(%rip)
	movw	$-1, 232+quote_calc2_check(%rip)
	movw	$-1, 234+quote_calc2_check(%rip)
	movw	$-1, 236+quote_calc2_check(%rip)
	movw	$124, 238+quote_calc2_check(%rip)
	movw	$124, 240+quote_calc2_check(%rip)
	movw	$-1, 242+quote_calc2_check(%rip)
	movw	$-1, 244+quote_calc2_check(%rip)
	movw	$-1, 246+quote_calc2_check(%rip)
	movw	$-1, 248+quote_calc2_check(%rip)
	movw	$-1, 250+quote_calc2_check(%rip)
	movw	$-1, 252+quote_calc2_check(%rip)
	movw	$-1, 254+quote_calc2_check(%rip)
	movw	$-1, 256+quote_calc2_check(%rip)
	movw	$-1, 258+quote_calc2_check(%rip)
	movw	$-1, 260+quote_calc2_check(%rip)
	movw	$-1, 262+quote_calc2_check(%rip)
	movw	$-1, 264+quote_calc2_check(%rip)
	movw	$-1, 266+quote_calc2_check(%rip)
	movw	$-1, 268+quote_calc2_check(%rip)
	movw	$-1, 270+quote_calc2_check(%rip)
	movw	$-1, 272+quote_calc2_check(%rip)
	movw	$258, 274+quote_calc2_check(%rip)
	movw	$-1, 276+quote_calc2_check(%rip)
	movw	$260, 278+quote_calc2_check(%rip)
	movw	$-1, 280+quote_calc2_check(%rip)
	movw	$262, 282+quote_calc2_check(%rip)
	movw	$-1, 284+quote_calc2_check(%rip)
	movw	$264, 286+quote_calc2_check(%rip)
	movw	$-1, 288+quote_calc2_check(%rip)
	movw	$266, 290+quote_calc2_check(%rip)
	movw	$-1, 292+quote_calc2_check(%rip)
	movw	$268, 294+quote_calc2_check(%rip)
	movw	$-1, 296+quote_calc2_check(%rip)
	movw	$-1, 298+quote_calc2_check(%rip)
	movw	$-1, 300+quote_calc2_check(%rip)
	movw	$-1, 302+quote_calc2_check(%rip)
	movw	$-1, 304+quote_calc2_check(%rip)
	movw	$-1, 306+quote_calc2_check(%rip)
	movw	$-1, 308+quote_calc2_check(%rip)
	movw	$-1, 310+quote_calc2_check(%rip)
	movw	$-1, 312+quote_calc2_check(%rip)
	movw	$-1, 314+quote_calc2_check(%rip)
	movw	$-1, 316+quote_calc2_check(%rip)
	movw	$-1, 318+quote_calc2_check(%rip)
	movw	$-1, 320+quote_calc2_check(%rip)
	movw	$-1, 322+quote_calc2_check(%rip)
	movw	$-1, 324+quote_calc2_check(%rip)
	movw	$-1, 326+quote_calc2_check(%rip)
	movw	$-1, 328+quote_calc2_check(%rip)
	movw	$-1, 330+quote_calc2_check(%rip)
	movw	$-1, 332+quote_calc2_check(%rip)
	movw	$-1, 334+quote_calc2_check(%rip)
	movw	$-1, 336+quote_calc2_check(%rip)
	movw	$-1, 338+quote_calc2_check(%rip)
	movw	$-1, 340+quote_calc2_check(%rip)
	movw	$-1, 342+quote_calc2_check(%rip)
	movw	$-1, 344+quote_calc2_check(%rip)
	movw	$-1, 346+quote_calc2_check(%rip)
	movw	$-1, 348+quote_calc2_check(%rip)
	movw	$-1, 350+quote_calc2_check(%rip)
	movw	$-1, 352+quote_calc2_check(%rip)
	movw	$-1, 354+quote_calc2_check(%rip)
	movw	$-1, 356+quote_calc2_check(%rip)
	movw	$-1, 358+quote_calc2_check(%rip)
	movw	$-1, 360+quote_calc2_check(%rip)
	movw	$-1, 362+quote_calc2_check(%rip)
	movw	$-1, 364+quote_calc2_check(%rip)
	movw	$-1, 366+quote_calc2_check(%rip)
	movw	$-1, 368+quote_calc2_check(%rip)
	movw	$-1, 370+quote_calc2_check(%rip)
	movw	$-1, 372+quote_calc2_check(%rip)
	movw	$-1, 374+quote_calc2_check(%rip)
	movw	$-1, 376+quote_calc2_check(%rip)
	movw	$-1, 378+quote_calc2_check(%rip)
	movw	$-1, 380+quote_calc2_check(%rip)
	movw	$-1, 382+quote_calc2_check(%rip)
	movw	$-1, 384+quote_calc2_check(%rip)
	movw	$-1, 386+quote_calc2_check(%rip)
	movw	$-1, 388+quote_calc2_check(%rip)
	movw	$-1, 390+quote_calc2_check(%rip)
	movw	$-1, 392+quote_calc2_check(%rip)
	movw	$-1, 394+quote_calc2_check(%rip)
	movw	$-1, 396+quote_calc2_check(%rip)
	movw	$-1, 398+quote_calc2_check(%rip)
	movw	$-1, 400+quote_calc2_check(%rip)
	movw	$-1, 402+quote_calc2_check(%rip)
	movw	$-1, 404+quote_calc2_check(%rip)
	movw	$-1, 406+quote_calc2_check(%rip)
	movw	$-1, 408+quote_calc2_check(%rip)
	movw	$-1, 410+quote_calc2_check(%rip)
	movw	$-1, 412+quote_calc2_check(%rip)
	movw	$-1, 414+quote_calc2_check(%rip)
	movw	$-1, 416+quote_calc2_check(%rip)
	movw	$-1, 418+quote_calc2_check(%rip)
	movw	$-1, 420+quote_calc2_check(%rip)
	movw	$-1, 422+quote_calc2_check(%rip)
	movw	$-1, 424+quote_calc2_check(%rip)
	movw	$-1, 426+quote_calc2_check(%rip)
	movw	$-1, 428+quote_calc2_check(%rip)
	movw	$-1, 430+quote_calc2_check(%rip)
	movw	$-1, 432+quote_calc2_check(%rip)
	movw	$-1, 434+quote_calc2_check(%rip)
	movw	$256, 436+quote_calc2_check(%rip)
	movw	$-1, 438+quote_calc2_check(%rip)
	movw	$-1, 440+quote_calc2_check(%rip)
	movw	$-1, 442+quote_calc2_check(%rip)
	movw	$260, 444+quote_calc2_check(%rip)
	movw	$-1, 446+quote_calc2_check(%rip)
	movw	$260, 448+quote_calc2_check(%rip)
	movw	$-1, 450+quote_calc2_check(%rip)
	movw	$-1, 452+quote_calc2_check(%rip)
	movw	$-1, 454+quote_calc2_check(%rip)
	movw	$-1, 456+quote_calc2_check(%rip)
	movw	$-1, 458+quote_calc2_check(%rip)
	movw	$-1, 460+quote_calc2_check(%rip)
	movw	$269, 462+quote_calc2_check(%rip)
	movw	$270, 464+quote_calc2_check(%rip)
	movw	$269, 466+quote_calc2_check(%rip)
	movw	$270, 468+quote_calc2_check(%rip)
	movw	$258, 470+quote_calc2_check(%rip)
	movw	$-1, 472+quote_calc2_check(%rip)
	movw	$260, 474+quote_calc2_check(%rip)
	movw	$-1, 476+quote_calc2_check(%rip)
	movw	$262, 478+quote_calc2_check(%rip)
	movw	$-1, 480+quote_calc2_check(%rip)
	movw	$264, 482+quote_calc2_check(%rip)
	movw	$-1, 484+quote_calc2_check(%rip)
	movw	$266, 486+quote_calc2_check(%rip)
	movw	$-1, 488+quote_calc2_check(%rip)
	movw	$268, 490+quote_calc2_check(%rip)
	movw	$-1, 492+quote_calc2_check(%rip)
	movw	$-1, 494+quote_calc2_check(%rip)
	movw	$258, 496+quote_calc2_check(%rip)
	movw	$258, 498+quote_calc2_check(%rip)
	movw	$260, 500+quote_calc2_check(%rip)
	movw	$260, 502+quote_calc2_check(%rip)
	movw	$262, 504+quote_calc2_check(%rip)
	movw	$262, 506+quote_calc2_check(%rip)
	movw	$264, 508+quote_calc2_check(%rip)
	movw	$264, 510+quote_calc2_check(%rip)
	movw	$266, 512+quote_calc2_check(%rip)
	movw	$266, 514+quote_calc2_check(%rip)
	movw	$268, 516+quote_calc2_check(%rip)
	movw	$268, 518+quote_calc2_check(%rip)
	nop
.L75:
	movw	$16, quote_calc2_table(%rip)
	movw	$15, 2+quote_calc2_table(%rip)
	movw	$6, 4+quote_calc2_table(%rip)
	movw	$22, 6+quote_calc2_table(%rip)
	movw	$6, 8+quote_calc2_table(%rip)
	movw	$14, 10+quote_calc2_table(%rip)
	movw	$13, 12+quote_calc2_table(%rip)
	movw	$7, 14+quote_calc2_table(%rip)
	movw	$8, 16+quote_calc2_table(%rip)
	movw	$9, 18+quote_calc2_table(%rip)
	movw	$13, 20+quote_calc2_table(%rip)
	movw	$10, 22+quote_calc2_table(%rip)
	movw	$11, 24+quote_calc2_table(%rip)
	movw	$12, 26+quote_calc2_table(%rip)
	movw	$10, 28+quote_calc2_table(%rip)
	movw	$16, 30+quote_calc2_table(%rip)
	movw	$15, 32+quote_calc2_table(%rip)
	movw	$17, 34+quote_calc2_table(%rip)
	movw	$25, 36+quote_calc2_table(%rip)
	movw	$18, 38+quote_calc2_table(%rip)
	movw	$23, 40+quote_calc2_table(%rip)
	movw	$19, 42+quote_calc2_table(%rip)
	movw	$4, 44+quote_calc2_table(%rip)
	movw	$20, 46+quote_calc2_table(%rip)
	movw	$5, 48+quote_calc2_table(%rip)
	movw	$21, 50+quote_calc2_table(%rip)
	movw	$0, 52+quote_calc2_table(%rip)
	movw	$0, 54+quote_calc2_table(%rip)
	movw	$0, 56+quote_calc2_table(%rip)
	movw	$0, 58+quote_calc2_table(%rip)
	movw	$0, 60+quote_calc2_table(%rip)
	movw	$16, 62+quote_calc2_table(%rip)
	movw	$0, 64+quote_calc2_table(%rip)
	movw	$0, 66+quote_calc2_table(%rip)
	movw	$0, 68+quote_calc2_table(%rip)
	movw	$0, 70+quote_calc2_table(%rip)
	movw	$14, 72+quote_calc2_table(%rip)
	movw	$13, 74+quote_calc2_table(%rip)
	movw	$7, 76+quote_calc2_table(%rip)
	movw	$8, 78+quote_calc2_table(%rip)
	movw	$9, 80+quote_calc2_table(%rip)
	movw	$0, 82+quote_calc2_table(%rip)
	movw	$10, 84+quote_calc2_table(%rip)
	movw	$11, 86+quote_calc2_table(%rip)
	movw	$12, 88+quote_calc2_table(%rip)
	movw	$12, 90+quote_calc2_table(%rip)
	movw	$0, 92+quote_calc2_table(%rip)
	movw	$0, 94+quote_calc2_table(%rip)
	movw	$14, 96+quote_calc2_table(%rip)
	movw	$0, 98+quote_calc2_table(%rip)
	movw	$0, 100+quote_calc2_table(%rip)
	movw	$0, 102+quote_calc2_table(%rip)
	movw	$0, 104+quote_calc2_table(%rip)
	movw	$0, 106+quote_calc2_table(%rip)
	movw	$0, 108+quote_calc2_table(%rip)
	movw	$24, 110+quote_calc2_table(%rip)
	movw	$0, 112+quote_calc2_table(%rip)
	movw	$0, 114+quote_calc2_table(%rip)
	movw	$26, 116+quote_calc2_table(%rip)
	movw	$27, 118+quote_calc2_table(%rip)
	movw	$28, 120+quote_calc2_table(%rip)
	movw	$29, 122+quote_calc2_table(%rip)
	movw	$30, 124+quote_calc2_table(%rip)
	movw	$31, 126+quote_calc2_table(%rip)
	movw	$32, 128+quote_calc2_table(%rip)
	movw	$0, 130+quote_calc2_table(%rip)
	movw	$0, 132+quote_calc2_table(%rip)
	movw	$0, 134+quote_calc2_table(%rip)
	movw	$0, 136+quote_calc2_table(%rip)
	movw	$0, 138+quote_calc2_table(%rip)
	movw	$0, 140+quote_calc2_table(%rip)
	movw	$0, 142+quote_calc2_table(%rip)
	movw	$0, 144+quote_calc2_table(%rip)
	movw	$0, 146+quote_calc2_table(%rip)
	movw	$0, 148+quote_calc2_table(%rip)
	movw	$0, 150+quote_calc2_table(%rip)
	movw	$0, 152+quote_calc2_table(%rip)
	movw	$0, 154+quote_calc2_table(%rip)
	movw	$0, 156+quote_calc2_table(%rip)
	movw	$0, 158+quote_calc2_table(%rip)
	movw	$0, 160+quote_calc2_table(%rip)
	movw	$0, 162+quote_calc2_table(%rip)
	movw	$0, 164+quote_calc2_table(%rip)
	movw	$0, 166+quote_calc2_table(%rip)
	movw	$0, 168+quote_calc2_table(%rip)
	movw	$0, 170+quote_calc2_table(%rip)
	movw	$0, 172+quote_calc2_table(%rip)
	movw	$0, 174+quote_calc2_table(%rip)
	movw	$0, 176+quote_calc2_table(%rip)
	movw	$0, 178+quote_calc2_table(%rip)
	movw	$0, 180+quote_calc2_table(%rip)
	movw	$0, 182+quote_calc2_table(%rip)
	movw	$0, 184+quote_calc2_table(%rip)
	movw	$0, 186+quote_calc2_table(%rip)
	movw	$0, 188+quote_calc2_table(%rip)
	movw	$0, 190+quote_calc2_table(%rip)
	movw	$0, 192+quote_calc2_table(%rip)
	movw	$0, 194+quote_calc2_table(%rip)
	movw	$0, 196+quote_calc2_table(%rip)
	movw	$0, 198+quote_calc2_table(%rip)
	movw	$0, 200+quote_calc2_table(%rip)
	movw	$22, 202+quote_calc2_table(%rip)
	movw	$0, 204+quote_calc2_table(%rip)
	movw	$0, 206+quote_calc2_table(%rip)
	movw	$0, 208+quote_calc2_table(%rip)
	movw	$0, 210+quote_calc2_table(%rip)
	movw	$0, 212+quote_calc2_table(%rip)
	movw	$0, 214+quote_calc2_table(%rip)
	movw	$0, 216+quote_calc2_table(%rip)
	movw	$0, 218+quote_calc2_table(%rip)
	movw	$0, 220+quote_calc2_table(%rip)
	movw	$0, 222+quote_calc2_table(%rip)
	movw	$0, 224+quote_calc2_table(%rip)
	movw	$0, 226+quote_calc2_table(%rip)
	movw	$16, 228+quote_calc2_table(%rip)
	movw	$15, 230+quote_calc2_table(%rip)
	movw	$0, 232+quote_calc2_table(%rip)
	movw	$0, 234+quote_calc2_table(%rip)
	movw	$0, 236+quote_calc2_table(%rip)
	movw	$14, 238+quote_calc2_table(%rip)
	movw	$13, 240+quote_calc2_table(%rip)
	movw	$0, 242+quote_calc2_table(%rip)
	movw	$0, 244+quote_calc2_table(%rip)
	movw	$0, 246+quote_calc2_table(%rip)
	movw	$0, 248+quote_calc2_table(%rip)
	movw	$0, 250+quote_calc2_table(%rip)
	movw	$0, 252+quote_calc2_table(%rip)
	movw	$0, 254+quote_calc2_table(%rip)
	movw	$0, 256+quote_calc2_table(%rip)
	movw	$0, 258+quote_calc2_table(%rip)
	movw	$0, 260+quote_calc2_table(%rip)
	movw	$0, 262+quote_calc2_table(%rip)
	movw	$0, 264+quote_calc2_table(%rip)
	movw	$0, 266+quote_calc2_table(%rip)
	movw	$0, 268+quote_calc2_table(%rip)
	movw	$0, 270+quote_calc2_table(%rip)
	movw	$0, 272+quote_calc2_table(%rip)
	movw	$16, 274+quote_calc2_table(%rip)
	movw	$0, 276+quote_calc2_table(%rip)
	movw	$17, 278+quote_calc2_table(%rip)
	movw	$0, 280+quote_calc2_table(%rip)
	movw	$18, 282+quote_calc2_table(%rip)
	movw	$0, 284+quote_calc2_table(%rip)
	movw	$19, 286+quote_calc2_table(%rip)
	movw	$0, 288+quote_calc2_table(%rip)
	movw	$20, 290+quote_calc2_table(%rip)
	movw	$0, 292+quote_calc2_table(%rip)
	movw	$21, 294+quote_calc2_table(%rip)
	movw	$0, 296+quote_calc2_table(%rip)
	movw	$0, 298+quote_calc2_table(%rip)
	movw	$0, 300+quote_calc2_table(%rip)
	movw	$0, 302+quote_calc2_table(%rip)
	movw	$0, 304+quote_calc2_table(%rip)
	movw	$0, 306+quote_calc2_table(%rip)
	movw	$0, 308+quote_calc2_table(%rip)
	movw	$0, 310+quote_calc2_table(%rip)
	movw	$0, 312+quote_calc2_table(%rip)
	movw	$0, 314+quote_calc2_table(%rip)
	movw	$0, 316+quote_calc2_table(%rip)
	movw	$0, 318+quote_calc2_table(%rip)
	movw	$0, 320+quote_calc2_table(%rip)
	movw	$0, 322+quote_calc2_table(%rip)
	movw	$0, 324+quote_calc2_table(%rip)
	movw	$0, 326+quote_calc2_table(%rip)
	movw	$0, 328+quote_calc2_table(%rip)
	movw	$0, 330+quote_calc2_table(%rip)
	movw	$0, 332+quote_calc2_table(%rip)
	movw	$0, 334+quote_calc2_table(%rip)
	movw	$0, 336+quote_calc2_table(%rip)
	movw	$0, 338+quote_calc2_table(%rip)
	movw	$0, 340+quote_calc2_table(%rip)
	movw	$0, 342+quote_calc2_table(%rip)
	movw	$0, 344+quote_calc2_table(%rip)
	movw	$0, 346+quote_calc2_table(%rip)
	movw	$0, 348+quote_calc2_table(%rip)
	movw	$0, 350+quote_calc2_table(%rip)
	movw	$0, 352+quote_calc2_table(%rip)
	movw	$0, 354+quote_calc2_table(%rip)
	movw	$0, 356+quote_calc2_table(%rip)
	movw	$0, 358+quote_calc2_table(%rip)
	movw	$0, 360+quote_calc2_table(%rip)
	movw	$0, 362+quote_calc2_table(%rip)
	movw	$0, 364+quote_calc2_table(%rip)
	movw	$0, 366+quote_calc2_table(%rip)
	movw	$0, 368+quote_calc2_table(%rip)
	movw	$0, 370+quote_calc2_table(%rip)
	movw	$0, 372+quote_calc2_table(%rip)
	movw	$0, 374+quote_calc2_table(%rip)
	movw	$0, 376+quote_calc2_table(%rip)
	movw	$0, 378+quote_calc2_table(%rip)
	movw	$0, 380+quote_calc2_table(%rip)
	movw	$0, 382+quote_calc2_table(%rip)
	movw	$0, 384+quote_calc2_table(%rip)
	movw	$0, 386+quote_calc2_table(%rip)
	movw	$0, 388+quote_calc2_table(%rip)
	movw	$0, 390+quote_calc2_table(%rip)
	movw	$0, 392+quote_calc2_table(%rip)
	movw	$0, 394+quote_calc2_table(%rip)
	movw	$0, 396+quote_calc2_table(%rip)
	movw	$0, 398+quote_calc2_table(%rip)
	movw	$0, 400+quote_calc2_table(%rip)
	movw	$0, 402+quote_calc2_table(%rip)
	movw	$0, 404+quote_calc2_table(%rip)
	movw	$0, 406+quote_calc2_table(%rip)
	movw	$0, 408+quote_calc2_table(%rip)
	movw	$0, 410+quote_calc2_table(%rip)
	movw	$0, 412+quote_calc2_table(%rip)
	movw	$0, 414+quote_calc2_table(%rip)
	movw	$0, 416+quote_calc2_table(%rip)
	movw	$0, 418+quote_calc2_table(%rip)
	movw	$0, 420+quote_calc2_table(%rip)
	movw	$0, 422+quote_calc2_table(%rip)
	movw	$0, 424+quote_calc2_table(%rip)
	movw	$0, 426+quote_calc2_table(%rip)
	movw	$0, 428+quote_calc2_table(%rip)
	movw	$0, 430+quote_calc2_table(%rip)
	movw	$0, 432+quote_calc2_table(%rip)
	movw	$0, 434+quote_calc2_table(%rip)
	movw	$2, 436+quote_calc2_table(%rip)
	movw	$0, 438+quote_calc2_table(%rip)
	movw	$0, 440+quote_calc2_table(%rip)
	movw	$0, 442+quote_calc2_table(%rip)
	movw	$3, 444+quote_calc2_table(%rip)
	movw	$0, 446+quote_calc2_table(%rip)
	movw	$3, 448+quote_calc2_table(%rip)
	movw	$0, 450+quote_calc2_table(%rip)
	movw	$0, 452+quote_calc2_table(%rip)
	movw	$0, 454+quote_calc2_table(%rip)
	movw	$0, 456+quote_calc2_table(%rip)
	movw	$0, 458+quote_calc2_table(%rip)
	movw	$0, 460+quote_calc2_table(%rip)
	movw	$4, 462+quote_calc2_table(%rip)
	movw	$5, 464+quote_calc2_table(%rip)
	movw	$4, 466+quote_calc2_table(%rip)
	movw	$11, 468+quote_calc2_table(%rip)
	movw	$16, 470+quote_calc2_table(%rip)
	movw	$0, 472+quote_calc2_table(%rip)
	movw	$17, 474+quote_calc2_table(%rip)
	movw	$0, 476+quote_calc2_table(%rip)
	movw	$18, 478+quote_calc2_table(%rip)
	movw	$0, 480+quote_calc2_table(%rip)
	movw	$19, 482+quote_calc2_table(%rip)
	movw	$0, 484+quote_calc2_table(%rip)
	movw	$20, 486+quote_calc2_table(%rip)
	movw	$0, 488+quote_calc2_table(%rip)
	movw	$21, 490+quote_calc2_table(%rip)
	movw	$0, 492+quote_calc2_table(%rip)
	movw	$0, 494+quote_calc2_table(%rip)
	movw	$16, 496+quote_calc2_table(%rip)
	movw	$15, 498+quote_calc2_table(%rip)
	movw	$16, 500+quote_calc2_table(%rip)
	movw	$15, 502+quote_calc2_table(%rip)
	movw	$16, 504+quote_calc2_table(%rip)
	movw	$15, 506+quote_calc2_table(%rip)
	movw	$16, 508+quote_calc2_table(%rip)
	movw	$15, 510+quote_calc2_table(%rip)
	movw	$16, 512+quote_calc2_table(%rip)
	movw	$15, 514+quote_calc2_table(%rip)
	movw	$16, 516+quote_calc2_table(%rip)
	movw	$15, 518+quote_calc2_table(%rip)
	nop
.L76:
	movw	$0, quote_calc2_gindex(%rip)
	movw	$0, 2+quote_calc2_gindex(%rip)
	movw	$42, 4+quote_calc2_gindex(%rip)
	movw	$0, 6+quote_calc2_gindex(%rip)
	nop
.L77:
	movw	$0, quote_calc2_rindex(%rip)
	movw	$0, 2+quote_calc2_rindex(%rip)
	movw	$0, 4+quote_calc2_rindex(%rip)
	movw	$0, 6+quote_calc2_rindex(%rip)
	movw	$0, 8+quote_calc2_rindex(%rip)
	movw	$-9, 10+quote_calc2_rindex(%rip)
	movw	$0, 12+quote_calc2_rindex(%rip)
	movw	$0, 14+quote_calc2_rindex(%rip)
	movw	$12, 16+quote_calc2_rindex(%rip)
	movw	$-10, 18+quote_calc2_rindex(%rip)
	movw	$0, 20+quote_calc2_rindex(%rip)
	movw	$0, 22+quote_calc2_rindex(%rip)
	movw	$-5, 24+quote_calc2_rindex(%rip)
	movw	$0, 26+quote_calc2_rindex(%rip)
	movw	$0, 28+quote_calc2_rindex(%rip)
	movw	$0, 30+quote_calc2_rindex(%rip)
	movw	$0, 32+quote_calc2_rindex(%rip)
	movw	$0, 34+quote_calc2_rindex(%rip)
	movw	$0, 36+quote_calc2_rindex(%rip)
	movw	$0, 38+quote_calc2_rindex(%rip)
	movw	$0, 40+quote_calc2_rindex(%rip)
	movw	$0, 42+quote_calc2_rindex(%rip)
	movw	$0, 44+quote_calc2_rindex(%rip)
	movw	$0, 46+quote_calc2_rindex(%rip)
	movw	$14, 48+quote_calc2_rindex(%rip)
	movw	$0, 50+quote_calc2_rindex(%rip)
	movw	$-3, 52+quote_calc2_rindex(%rip)
	movw	$-2, 54+quote_calc2_rindex(%rip)
	movw	$-1, 56+quote_calc2_rindex(%rip)
	movw	$1, 58+quote_calc2_rindex(%rip)
	movw	$2, 60+quote_calc2_rindex(%rip)
	movw	$3, 62+quote_calc2_rindex(%rip)
	movw	$-4, 64+quote_calc2_rindex(%rip)
	nop
.L78:
	movw	$0, quote_calc2_sindex(%rip)
	movw	$-38, 2+quote_calc2_sindex(%rip)
	movw	$4, 4+quote_calc2_sindex(%rip)
	movw	$-36, 6+quote_calc2_sindex(%rip)
	movw	$0, 8+quote_calc2_sindex(%rip)
	movw	$-51, 10+quote_calc2_sindex(%rip)
	movw	$-36, 12+quote_calc2_sindex(%rip)
	movw	$6, 14+quote_calc2_sindex(%rip)
	movw	$-121, 16+quote_calc2_sindex(%rip)
	movw	$-249, 18+quote_calc2_sindex(%rip)
	movw	$0, 20+quote_calc2_sindex(%rip)
	movw	$0, 22+quote_calc2_sindex(%rip)
	movw	$-243, 24+quote_calc2_sindex(%rip)
	movw	$-36, 26+quote_calc2_sindex(%rip)
	movw	$-23, 28+quote_calc2_sindex(%rip)
	movw	$0, 30+quote_calc2_sindex(%rip)
	movw	$-36, 32+quote_calc2_sindex(%rip)
	movw	$-36, 34+quote_calc2_sindex(%rip)
	movw	$-36, 36+quote_calc2_sindex(%rip)
	movw	$-36, 38+quote_calc2_sindex(%rip)
	movw	$-36, 40+quote_calc2_sindex(%rip)
	movw	$-36, 42+quote_calc2_sindex(%rip)
	movw	$-36, 44+quote_calc2_sindex(%rip)
	movw	$0, 46+quote_calc2_sindex(%rip)
	movw	$-121, 48+quote_calc2_sindex(%rip)
	movw	$0, 50+quote_calc2_sindex(%rip)
	movw	$-121, 52+quote_calc2_sindex(%rip)
	movw	$-121, 54+quote_calc2_sindex(%rip)
	movw	$-121, 56+quote_calc2_sindex(%rip)
	movw	$-121, 58+quote_calc2_sindex(%rip)
	movw	$-121, 60+quote_calc2_sindex(%rip)
	movw	$-121, 62+quote_calc2_sindex(%rip)
	movw	$-243, 64+quote_calc2_sindex(%rip)
	nop
.L79:
	movw	$1, quote_calc2_dgoto(%rip)
	movw	$7, 2+quote_calc2_dgoto(%rip)
	movw	$8, 4+quote_calc2_dgoto(%rip)
	movw	$9, 6+quote_calc2_dgoto(%rip)
	nop
.L80:
	movw	$1, quote_calc2_defred(%rip)
	movw	$0, 2+quote_calc2_defred(%rip)
	movw	$0, 4+quote_calc2_defred(%rip)
	movw	$0, 6+quote_calc2_defred(%rip)
	movw	$17, 8+quote_calc2_defred(%rip)
	movw	$0, 10+quote_calc2_defred(%rip)
	movw	$0, 12+quote_calc2_defred(%rip)
	movw	$0, 14+quote_calc2_defred(%rip)
	movw	$0, 16+quote_calc2_defred(%rip)
	movw	$0, 18+quote_calc2_defred(%rip)
	movw	$3, 20+quote_calc2_defred(%rip)
	movw	$15, 22+quote_calc2_defred(%rip)
	movw	$0, 24+quote_calc2_defred(%rip)
	movw	$0, 26+quote_calc2_defred(%rip)
	movw	$0, 28+quote_calc2_defred(%rip)
	movw	$2, 30+quote_calc2_defred(%rip)
	movw	$0, 32+quote_calc2_defred(%rip)
	movw	$0, 34+quote_calc2_defred(%rip)
	movw	$0, 36+quote_calc2_defred(%rip)
	movw	$0, 38+quote_calc2_defred(%rip)
	movw	$0, 40+quote_calc2_defred(%rip)
	movw	$0, 42+quote_calc2_defred(%rip)
	movw	$0, 44+quote_calc2_defred(%rip)
	movw	$18, 46+quote_calc2_defred(%rip)
	movw	$0, 48+quote_calc2_defred(%rip)
	movw	$6, 50+quote_calc2_defred(%rip)
	movw	$0, 52+quote_calc2_defred(%rip)
	movw	$0, 54+quote_calc2_defred(%rip)
	movw	$0, 56+quote_calc2_defred(%rip)
	movw	$0, 58+quote_calc2_defred(%rip)
	movw	$0, 60+quote_calc2_defred(%rip)
	movw	$0, 62+quote_calc2_defred(%rip)
	movw	$0, 64+quote_calc2_defred(%rip)
	nop
.L81:
	movw	$2, quote_calc2_len(%rip)
	movw	$0, 2+quote_calc2_len(%rip)
	movw	$3, 4+quote_calc2_len(%rip)
	movw	$3, 6+quote_calc2_len(%rip)
	movw	$1, 8+quote_calc2_len(%rip)
	movw	$3, 10+quote_calc2_len(%rip)
	movw	$3, 12+quote_calc2_len(%rip)
	movw	$3, 14+quote_calc2_len(%rip)
	movw	$3, 16+quote_calc2_len(%rip)
	movw	$3, 18+quote_calc2_len(%rip)
	movw	$3, 20+quote_calc2_len(%rip)
	movw	$3, 22+quote_calc2_len(%rip)
	movw	$3, 24+quote_calc2_len(%rip)
	movw	$3, 26+quote_calc2_len(%rip)
	movw	$2, 28+quote_calc2_len(%rip)
	movw	$1, 30+quote_calc2_len(%rip)
	movw	$1, 32+quote_calc2_len(%rip)
	movw	$1, 34+quote_calc2_len(%rip)
	movw	$2, 36+quote_calc2_len(%rip)
	nop
.L82:
	movw	$-1, quote_calc2_lhs(%rip)
	movw	$0, 2+quote_calc2_lhs(%rip)
	movw	$0, 4+quote_calc2_lhs(%rip)
	movw	$0, 6+quote_calc2_lhs(%rip)
	movw	$1, 8+quote_calc2_lhs(%rip)
	movw	$1, 10+quote_calc2_lhs(%rip)
	movw	$2, 12+quote_calc2_lhs(%rip)
	movw	$2, 14+quote_calc2_lhs(%rip)
	movw	$2, 16+quote_calc2_lhs(%rip)
	movw	$2, 18+quote_calc2_lhs(%rip)
	movw	$2, 20+quote_calc2_lhs(%rip)
	movw	$2, 22+quote_calc2_lhs(%rip)
	movw	$2, 24+quote_calc2_lhs(%rip)
	movw	$2, 26+quote_calc2_lhs(%rip)
	movw	$2, 28+quote_calc2_lhs(%rip)
	movw	$2, 30+quote_calc2_lhs(%rip)
	movw	$2, 32+quote_calc2_lhs(%rip)
	movw	$3, 34+quote_calc2_lhs(%rip)
	movw	$3, 36+quote_calc2_lhs(%rip)
	nop
.L83:
	movl	$0, base(%rip)
	nop
.L84:
	movl	$0, -16(%rbp)
	jmp	.L85
.L86:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	regs(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -16(%rbp)
.L85:
	cmpl	$25, -16(%rbp)
	jle	.L86
	nop
.L87:
	movb	$64, yysccsid(%rip)
	movb	$40, 1+yysccsid(%rip)
	movb	$35, 2+yysccsid(%rip)
	movb	$41, 3+yysccsid(%rip)
	movb	$121, 4+yysccsid(%rip)
	movb	$97, 5+yysccsid(%rip)
	movb	$99, 6+yysccsid(%rip)
	movb	$99, 7+yysccsid(%rip)
	movb	$112, 8+yysccsid(%rip)
	movb	$97, 9+yysccsid(%rip)
	movb	$114, 10+yysccsid(%rip)
	movb	$9, 11+yysccsid(%rip)
	movb	$49, 12+yysccsid(%rip)
	movb	$46, 13+yysccsid(%rip)
	movb	$57, 14+yysccsid(%rip)
	movb	$32, 15+yysccsid(%rip)
	movb	$40, 16+yysccsid(%rip)
	movb	$66, 17+yysccsid(%rip)
	movb	$101, 18+yysccsid(%rip)
	movb	$114, 19+yysccsid(%rip)
	movb	$107, 20+yysccsid(%rip)
	movb	$101, 21+yysccsid(%rip)
	movb	$108, 22+yysccsid(%rip)
	movb	$101, 23+yysccsid(%rip)
	movb	$121, 24+yysccsid(%rip)
	movb	$41, 25+yysccsid(%rip)
	movb	$32, 26+yysccsid(%rip)
	movb	$48, 27+yysccsid(%rip)
	movb	$50, 28+yysccsid(%rip)
	movb	$47, 29+yysccsid(%rip)
	movb	$50, 30+yysccsid(%rip)
	movb	$49, 31+yysccsid(%rip)
	movb	$47, 32+yysccsid(%rip)
	movb	$57, 33+yysccsid(%rip)
	movb	$51, 34+yysccsid(%rip)
	movb	$0, 35+yysccsid(%rip)
	nop
.L88:
	movq	$0, _TIG_IZ_EK2N_envp(%rip)
	nop
.L89:
	movq	$0, _TIG_IZ_EK2N_argv(%rip)
	nop
.L90:
	movl	$0, _TIG_IZ_EK2N_argc(%rip)
	nop
	nop
.L91:
.L92:
#APP
# 970 "quote_calc2.y" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EK2N--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_EK2N_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_EK2N_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_EK2N_envp(%rip)
	nop
	movq	$5, -8(%rbp)
.L104:
	cmpq	$5, -8(%rbp)
	ja	.L106
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L95(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L95(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L95:
	.long	.L99-.L95
	.long	.L106-.L95
	.long	.L98-.L95
	.long	.L97-.L95
	.long	.L96-.L95
	.long	.L94-.L95
	.text
.L96:
	cmpl	$0, -12(%rbp)
	je	.L100
	movq	$0, -8(%rbp)
	jmp	.L102
.L100:
	movq	$2, -8(%rbp)
	jmp	.L102
.L97:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L102
.L94:
	movq	$3, -8(%rbp)
	jmp	.L102
.L99:
	movl	$0, %eax
	jmp	.L105
.L98:
	call	quote_calc2_parse
	movq	$3, -8(%rbp)
	jmp	.L102
.L106:
	nop
.L102:
	jmp	.L104
.L105:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC1:
	.string	"syntax error"
.LC2:
	.string	"%d\n"
.LC3:
	.string	"yacc stack overflow"
	.text
	.globl	quote_calc2_parse
	.type	quote_calc2_parse, @function
quote_calc2_parse:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	$38, -8(%rbp)
.L313:
	cmpq	$144, -8(%rbp)
	ja	.L314
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L110(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L110(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L110:
	.long	.L314-.L110
	.long	.L212-.L110
	.long	.L211-.L110
	.long	.L210-.L110
	.long	.L314-.L110
	.long	.L209-.L110
	.long	.L208-.L110
	.long	.L207-.L110
	.long	.L206-.L110
	.long	.L205-.L110
	.long	.L204-.L110
	.long	.L203-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L202-.L110
	.long	.L201-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L200-.L110
	.long	.L199-.L110
	.long	.L198-.L110
	.long	.L314-.L110
	.long	.L197-.L110
	.long	.L196-.L110
	.long	.L195-.L110
	.long	.L194-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L193-.L110
	.long	.L192-.L110
	.long	.L314-.L110
	.long	.L191-.L110
	.long	.L190-.L110
	.long	.L314-.L110
	.long	.L189-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L188-.L110
	.long	.L187-.L110
	.long	.L186-.L110
	.long	.L314-.L110
	.long	.L185-.L110
	.long	.L184-.L110
	.long	.L183-.L110
	.long	.L182-.L110
	.long	.L181-.L110
	.long	.L180-.L110
	.long	.L179-.L110
	.long	.L178-.L110
	.long	.L177-.L110
	.long	.L314-.L110
	.long	.L176-.L110
	.long	.L175-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L174-.L110
	.long	.L173-.L110
	.long	.L172-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L171-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L170-.L110
	.long	.L314-.L110
	.long	.L169-.L110
	.long	.L168-.L110
	.long	.L314-.L110
	.long	.L167-.L110
	.long	.L166-.L110
	.long	.L165-.L110
	.long	.L164-.L110
	.long	.L163-.L110
	.long	.L162-.L110
	.long	.L314-.L110
	.long	.L161-.L110
	.long	.L160-.L110
	.long	.L314-.L110
	.long	.L159-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L158-.L110
	.long	.L157-.L110
	.long	.L156-.L110
	.long	.L314-.L110
	.long	.L155-.L110
	.long	.L154-.L110
	.long	.L153-.L110
	.long	.L152-.L110
	.long	.L314-.L110
	.long	.L314-.L110
	.long	.L151-.L110
	.long	.L150-.L110
	.long	.L149-.L110
	.long	.L314-.L110
	.long	.L148-.L110
	.long	.L147-.L110
	.long	.L146-.L110
	.long	.L145-.L110
	.long	.L144-.L110
	.long	.L143-.L110
	.long	.L142-.L110
	.long	.L141-.L110
	.long	.L140-.L110
	.long	.L314-.L110
	.long	.L139-.L110
	.long	.L314-.L110
	.long	.L138-.L110
	.long	.L137-.L110
	.long	.L136-.L110
	.long	.L314-.L110
	.long	.L135-.L110
	.long	.L134-.L110
	.long	.L133-.L110
	.long	.L132-.L110
	.long	.L131-.L110
	.long	.L314-.L110
	.long	.L130-.L110
	.long	.L129-.L110
	.long	.L128-.L110
	.long	.L127-.L110
	.long	.L126-.L110
	.long	.L125-.L110
	.long	.L124-.L110
	.long	.L123-.L110
	.long	.L122-.L110
	.long	.L233-.L110
	.long	.L314-.L110
	.long	.L120-.L110
	.long	.L119-.L110
	.long	.L118-.L110
	.long	.L117-.L110
	.long	.L116-.L110
	.long	.L314-.L110
	.long	.L115-.L110
	.long	.L114-.L110
	.long	.L113-.L110
	.long	.L112-.L110
	.long	.L111-.L110
	.long	.L314-.L110
	.long	.L109-.L110
	.text
.L200:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_len(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -36(%rbp)
	movq	$136, -8(%rbp)
	jmp	.L213
.L122:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %eax
	movq	40+yystack(%rip), %rdx
	movl	(%rdx), %esi
	cltd
	idivl	%esi
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L177:
	movq	40+yystack(%rip), %rax
	movq	40+yystack(%rip), %rdx
	subq	$8, %rdx
	movl	(%rdx), %edx
	movl	(%rax), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	regs(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$142, -8(%rbp)
	jmp	.L213
.L143:
	movl	-32(%rbp), %eax
	subl	$3, %eax
	cmpl	$15, %eax
	ja	.L214
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L216(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L216(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L216:
	.long	.L230-.L216
	.long	.L229-.L216
	.long	.L228-.L216
	.long	.L227-.L216
	.long	.L226-.L216
	.long	.L225-.L216
	.long	.L224-.L216
	.long	.L223-.L216
	.long	.L222-.L216
	.long	.L221-.L216
	.long	.L220-.L216
	.long	.L219-.L216
	.long	.L218-.L216
	.long	.L214-.L216
	.long	.L217-.L216
	.long	.L215-.L216
	.text
.L215:
	movq	$32, -8(%rbp)
	jmp	.L231
.L217:
	movq	$112, -8(%rbp)
	jmp	.L231
.L218:
	movq	$76, -8(%rbp)
	jmp	.L231
.L219:
	movq	$128, -8(%rbp)
	jmp	.L231
.L220:
	movq	$61, -8(%rbp)
	jmp	.L231
.L221:
	movq	$56, -8(%rbp)
	jmp	.L231
.L222:
	movq	$116, -8(%rbp)
	jmp	.L231
.L223:
	movq	$129, -8(%rbp)
	jmp	.L231
.L224:
	movq	$40, -8(%rbp)
	jmp	.L231
.L225:
	movq	$87, -8(%rbp)
	jmp	.L231
.L226:
	movq	$47, -8(%rbp)
	jmp	.L231
.L227:
	movq	$90, -8(%rbp)
	jmp	.L231
.L228:
	movq	$50, -8(%rbp)
	jmp	.L231
.L229:
	movq	$97, -8(%rbp)
	jmp	.L231
.L230:
	movq	$7, -8(%rbp)
	jmp	.L231
.L214:
	movq	$122, -8(%rbp)
	nop
.L231:
	jmp	.L213
.L121:
.L232:
.L233:
	movq	16+yystack(%rip), %rax
	movq	%rax, %rdx
	movq	8+yystack(%rip), %rax
	cmpq	%rax, %rdx
	ja	.L234
	movq	$99, -8(%rbp)
	jmp	.L213
.L234:
	movq	$118, -8(%rbp)
	jmp	.L213
.L194:
	cmpl	$0, -32(%rbp)
	je	.L236
	movq	$46, -8(%rbp)
	jmp	.L213
.L236:
	movq	$24, -8(%rbp)
	jmp	.L213
.L178:
	leaq	yystack(%rip), %rax
	movq	%rax, %rdi
	call	yygrowstack
	movl	%eax, -20(%rbp)
	movq	$95, -8(%rbp)
	jmp	.L213
.L176:
	movl	-28(%rbp), %eax
	addl	%eax, -32(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L213
.L192:
	movl	$3, quote_calc2_errflag(%rip)
	movq	$48, -8(%rbp)
	jmp	.L213
.L144:
	cmpl	$0, -32(%rbp)
	je	.L238
	movq	$89, -8(%rbp)
	jmp	.L213
.L238:
	movq	$44, -8(%rbp)
	jmp	.L213
.L145:
	cmpl	$259, -32(%rbp)
	jg	.L240
	movq	$101, -8(%rbp)
	jmp	.L213
.L240:
	movq	$132, -8(%rbp)
	jmp	.L213
.L141:
	cmpl	$259, -32(%rbp)
	jg	.L242
	movq	$71, -8(%rbp)
	jmp	.L213
.L242:
	movq	$130, -8(%rbp)
	jmp	.L213
.L202:
	movl	quote_calc2_char(%rip), %eax
	testl	%eax, %eax
	jns	.L244
	movq	$127, -8(%rbp)
	jmp	.L213
.L244:
	movq	$66, -8(%rbp)
	jmp	.L213
.L138:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_table(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	16+yystack(%rip), %rax
	addq	$2, %rax
	movq	%rax, 16+yystack(%rip)
	movq	16+yystack(%rip), %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	(%rdx,%rdx), %rcx
	leaq	quote_calc2_table(%rip), %rdx
	movzwl	(%rcx,%rdx), %edx
	movw	%dx, (%rax)
	movq	40+yystack(%rip), %rax
	addq	$4, %rax
	movq	%rax, 40+yystack(%rip)
	movq	40+yystack(%rip), %rax
	movl	quote_calc2_lval(%rip), %edx
	movl	%edx, (%rax)
	movl	$-1, quote_calc2_char(%rip)
	movq	$123, -8(%rbp)
	jmp	.L213
.L201:
	movl	$0, quote_calc2_char(%rip)
	movq	$139, -8(%rbp)
	jmp	.L213
.L119:
	movq	$29, -8(%rbp)
	jmp	.L213
.L155:
	movl	quote_calc2_char(%rip), %eax
	addl	%eax, -32(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L213
.L174:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %edx
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	andl	%edx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L159:
	movl	$0, quote_calc2_char(%rip)
	movq	$66, -8(%rbp)
	jmp	.L213
.L130:
	movl	quote_calc2_char(%rip), %eax
	testl	%eax, %eax
	jns	.L246
	movq	$15, -8(%rbp)
	jmp	.L213
.L246:
	movq	$139, -8(%rbp)
	jmp	.L213
.L112:
	cmpl	$0, -32(%rbp)
	js	.L248
	movq	$73, -8(%rbp)
	jmp	.L213
.L248:
	movq	$24, -8(%rbp)
	jmp	.L213
.L146:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_check(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	cmpl	%eax, -28(%rbp)
	jne	.L250
	movq	$115, -8(%rbp)
	jmp	.L213
.L250:
	movq	$45, -8(%rbp)
	jmp	.L213
.L167:
	cmpl	$0, -36(%rbp)
	jne	.L252
	movq	$109, -8(%rbp)
	jmp	.L213
.L252:
	movq	$22, -8(%rbp)
	jmp	.L213
.L206:
	movl	$8, base(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L150:
	cmpl	$0, -32(%rbp)
	je	.L254
	movq	$18, -8(%rbp)
	jmp	.L213
.L254:
	movq	$77, -8(%rbp)
	jmp	.L213
.L182:
	movl	-36(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_dgoto(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	$144, -8(%rbp)
	jmp	.L213
.L127:
	movl	quote_calc2_nerrs(%rip), %eax
	addl	$1, %eax
	movl	%eax, quote_calc2_nerrs(%rip)
	movq	$53, -8(%rbp)
	jmp	.L213
.L132:
	movq	16+yystack(%rip), %rax
	subq	$2, %rax
	movq	%rax, 16+yystack(%rip)
	movq	40+yystack(%rip), %rax
	subq	$4, %rax
	movq	%rax, 40+yystack(%rip)
	movq	$48, -8(%rbp)
	jmp	.L213
.L113:
	cmpl	$0, -24(%rbp)
	je	.L256
	movq	$43, -8(%rbp)
	jmp	.L213
.L256:
	movq	$86, -8(%rbp)
	jmp	.L213
.L212:
	movl	$10, base(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L123:
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	negl	%eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L196:
	movq	40+yystack(%rip), %rdx
	movl	$1, %eax
	subl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$104, -8(%rbp)
	jmp	.L213
.L160:
	movl	quote_calc2_char(%rip), %eax
	testl	%eax, %eax
	jns	.L258
	movq	$72, -8(%rbp)
	jmp	.L213
.L258:
	movq	$139, -8(%rbp)
	jmp	.L213
.L166:
	movq	8+yystack(%rip), %rax
	testq	%rax, %rax
	jne	.L260
	movq	$134, -8(%rbp)
	jmp	.L213
.L260:
	movq	$86, -8(%rbp)
	jmp	.L213
.L210:
	cmpl	$0, -28(%rbp)
	jne	.L262
	movq	$69, -8(%rbp)
	jmp	.L213
.L262:
	movq	$22, -8(%rbp)
	jmp	.L213
.L118:
	leaq	yystack(%rip), %rax
	movq	%rax, %rdi
	call	yygrowstack
	movl	%eax, -24(%rbp)
	movq	$140, -8(%rbp)
	jmp	.L213
.L195:
	movl	-28(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_rindex(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -32(%rbp)
	movq	$103, -8(%rbp)
	jmp	.L213
.L139:
	movl	$1, -28(%rbp)
	movq	16+yystack(%rip), %rax
	addq	$2, %rax
	movq	%rax, 16+yystack(%rip)
	movq	16+yystack(%rip), %rax
	movw	$1, (%rax)
	movq	40+yystack(%rip), %rax
	addq	$4, %rax
	movq	%rax, 40+yystack(%rip)
	movq	40+yystack(%rip), %rax
	movl	quote_calc2_val(%rip), %edx
	movl	%edx, (%rax)
	movq	$14, -8(%rbp)
	jmp	.L213
.L161:
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	regs(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L173:
	leaq	yystack(%rip), %rax
	movq	%rax, %rdi
	call	yygrowstack
	movl	%eax, -12(%rbp)
	movq	$85, -8(%rbp)
	jmp	.L213
.L129:
	movq	$142, -8(%rbp)
	jmp	.L213
.L158:
	cmpl	$0, -12(%rbp)
	je	.L264
	movq	$43, -8(%rbp)
	jmp	.L213
.L264:
	movq	$67, -8(%rbp)
	jmp	.L213
.L147:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_check(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	movswl	%ax, %edx
	movl	quote_calc2_char(%rip), %eax
	cmpl	%eax, %edx
	jne	.L266
	movq	$74, -8(%rbp)
	jmp	.L213
.L266:
	movq	$44, -8(%rbp)
	jmp	.L213
.L142:
	cmpl	$259, -32(%rbp)
	jg	.L268
	movq	$100, -8(%rbp)
	jmp	.L213
.L268:
	movq	$44, -8(%rbp)
	jmp	.L213
.L203:
	cmpl	$0, -32(%rbp)
	js	.L270
	movq	$102, -8(%rbp)
	jmp	.L213
.L270:
	movq	$107, -8(%rbp)
	jmp	.L213
.L205:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_check(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	movswl	%ax, %edx
	movl	quote_calc2_char(%rip), %eax
	cmpl	%eax, %edx
	jne	.L272
	movq	$20, -8(%rbp)
	jmp	.L213
.L272:
	movq	$24, -8(%rbp)
	jmp	.L213
.L117:
	leaq	yystack(%rip), %rax
	movq	%rax, %rdi
	call	yygrowstack
	movl	%eax, -16(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L213
.L140:
	movl	-36(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_dgoto(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	$144, -8(%rbp)
	jmp	.L213
.L126:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	quote_calc2_error
	movq	$124, -8(%rbp)
	jmp	.L213
.L199:
	cmpl	$0, -16(%rbp)
	je	.L274
	movq	$43, -8(%rbp)
	jmp	.L213
.L274:
	movq	$39, -8(%rbp)
	jmp	.L213
.L191:
	movq	40+yystack(%rip), %rax
	subq	$4, %rax
	movl	(%rax), %edx
	movl	base(%rip), %eax
	imull	%eax, %edx
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L154:
	movq	40+yystack(%rip), %rax
	movl	-4(%rax), %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L186:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %edx
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	imull	%edx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L168:
	movq	16+yystack(%rip), %rax
	addq	$2, %rax
	movq	%rax, 16+yystack(%rip)
	movq	16+yystack(%rip), %rax
	movl	-28(%rbp), %edx
	movw	%dx, (%rax)
	movq	40+yystack(%rip), %rax
	addq	$4, %rax
	movq	%rax, 40+yystack(%rip)
	movq	40+yystack(%rip), %rax
	movl	quote_calc2_val(%rip), %edx
	movl	%edx, (%rax)
	movq	$33, -8(%rbp)
	jmp	.L213
.L133:
	cmpl	$0, -32(%rbp)
	js	.L276
	movq	$106, -8(%rbp)
	jmp	.L213
.L276:
	movq	$130, -8(%rbp)
	jmp	.L213
.L120:
	movl	-36(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_dgoto(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	$144, -8(%rbp)
	jmp	.L213
.L208:
	cmpl	$0, -32(%rbp)
	je	.L278
	movq	$52, -8(%rbp)
	jmp	.L213
.L278:
	movq	$2, -8(%rbp)
	jmp	.L213
.L111:
	movq	16+yystack(%rip), %rdx
	movl	-36(%rbp), %eax
	cltq
	addq	%rax, %rax
	negq	%rax
	addq	%rdx, %rax
	movq	%rax, 16+yystack(%rip)
	movq	16+yystack(%rip), %rax
	movzwl	(%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	40+yystack(%rip), %rdx
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	negq	%rax
	addq	%rdx, %rax
	movq	%rax, 40+yystack(%rip)
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_lhs(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -36(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L213
.L109:
	movq	16+yystack(%rip), %rax
	movq	%rax, %rdx
	movq	24+yystack(%rip), %rax
	cmpq	%rax, %rdx
	jb	.L280
	movq	$57, -8(%rbp)
	jmp	.L213
.L280:
	movq	$67, -8(%rbp)
	jmp	.L213
.L134:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %eax
	movq	40+yystack(%rip), %rdx
	movl	(%rdx), %ecx
	cltd
	idivl	%ecx
	movl	%edx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L188:
	movq	$91, -8(%rbp)
	jmp	.L213
.L171:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %edx
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	orl	%edx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L156:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %edx
	movq	40+yystack(%rip), %rax
	movl	(%rax), %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L136:
	addl	$256, -32(%rbp)
	movq	$117, -8(%rbp)
	jmp	.L213
.L115:
	cmpl	$0, -32(%rbp)
	je	.L282
	movq	$113, -8(%rbp)
	jmp	.L213
.L282:
	movq	$130, -8(%rbp)
	jmp	.L213
.L172:
	movl	$0, quote_calc2_val(%rip)
	movq	$104, -8(%rbp)
	jmp	.L213
.L162:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_table(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -32(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L213
.L137:
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$64, -8(%rbp)
	jmp	.L213
.L128:
	movl	quote_calc2_errflag(%rip), %eax
	testl	%eax, %eax
	jle	.L284
	movq	$92, -8(%rbp)
	jmp	.L213
.L284:
	movq	$33, -8(%rbp)
	jmp	.L213
.L179:
	movq	16+yystack(%rip), %rax
	movzwl	(%rax), %eax
	cwtl
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_sindex(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -32(%rbp)
	movq	$138, -8(%rbp)
	jmp	.L213
.L124:
	call	quote_calc2_lex
	movl	%eax, quote_calc2_char(%rip)
	movq	$42, -8(%rbp)
	jmp	.L213
.L114:
	movl	-28(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_sindex(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -32(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L213
.L165:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_check(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cmpw	$256, %ax
	jne	.L286
	movq	$5, -8(%rbp)
	jmp	.L213
.L286:
	movq	$130, -8(%rbp)
	jmp	.L213
.L197:
	movl	-36(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_gindex(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -32(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L213
.L175:
	movl	quote_calc2_errflag(%rip), %eax
	cmpl	$2, %eax
	jg	.L288
	movq	$30, -8(%rbp)
	jmp	.L213
.L288:
	movq	$35, -8(%rbp)
	jmp	.L213
.L180:
	movq	40+yystack(%rip), %rax
	subq	$8, %rax
	movl	(%rax), %edx
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, quote_calc2_val(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L163:
	cmpl	$259, -32(%rbp)
	jg	.L290
	movq	$9, -8(%rbp)
	jmp	.L213
.L290:
	movq	$24, -8(%rbp)
	jmp	.L213
.L183:
	movl	quote_calc2_errflag(%rip), %eax
	testl	%eax, %eax
	je	.L292
	movq	$53, -8(%rbp)
	jmp	.L213
.L292:
	movq	$125, -8(%rbp)
	jmp	.L213
.L209:
	movq	16+yystack(%rip), %rax
	movq	%rax, %rdx
	movq	24+yystack(%rip), %rax
	cmpq	%rax, %rdx
	jb	.L294
	movq	$135, -8(%rbp)
	jmp	.L213
.L294:
	movq	$39, -8(%rbp)
	jmp	.L213
.L153:
	movl	$0, quote_calc2_nerrs(%rip)
	movl	$0, quote_calc2_errflag(%rip)
	movl	$-1, quote_calc2_char(%rip)
	movl	$0, -28(%rbp)
	movq	$70, -8(%rbp)
	jmp	.L213
.L149:
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$142, -8(%rbp)
	jmp	.L213
.L164:
	call	quote_calc2_lex
	movl	%eax, quote_calc2_char(%rip)
	movq	$121, -8(%rbp)
	jmp	.L213
.L148:
	movq	$119, -8(%rbp)
	jmp	.L213
.L190:
	movl	-28(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_defred(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -32(%rbp)
	movq	$96, -8(%rbp)
	jmp	.L213
.L170:
	movq	40+yystack(%rip), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L296
	movq	$8, -8(%rbp)
	jmp	.L213
.L296:
	movq	$1, -8(%rbp)
	jmp	.L213
.L131:
	movl	$1, %eax
	jmp	.L298
.L151:
	cmpl	$0, -20(%rbp)
	je	.L299
	movq	$43, -8(%rbp)
	jmp	.L213
.L299:
	movq	$111, -8(%rbp)
	jmp	.L213
.L152:
	movl	quote_calc2_errflag(%rip), %eax
	subl	$1, %eax
	movl	%eax, quote_calc2_errflag(%rip)
	movq	$33, -8(%rbp)
	jmp	.L213
.L135:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_table(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	$144, -8(%rbp)
	jmp	.L213
.L204:
	cmpl	$0, -32(%rbp)
	js	.L301
	movq	$105, -8(%rbp)
	jmp	.L213
.L301:
	movq	$44, -8(%rbp)
	jmp	.L213
.L185:
	movl	quote_calc2_char(%rip), %eax
	testl	%eax, %eax
	jns	.L303
	movq	$79, -8(%rbp)
	jmp	.L213
.L303:
	movq	$66, -8(%rbp)
	jmp	.L213
.L116:
	cmpl	$0, -36(%rbp)
	je	.L305
	movq	$23, -8(%rbp)
	jmp	.L213
.L305:
	movq	$58, -8(%rbp)
	jmp	.L213
.L181:
	movl	quote_calc2_char(%rip), %eax
	addl	%eax, -32(%rbp)
	movq	$141, -8(%rbp)
	jmp	.L213
.L187:
	movl	-32(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_table(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	16+yystack(%rip), %rax
	addq	$2, %rax
	movq	%rax, 16+yystack(%rip)
	movq	16+yystack(%rip), %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	(%rdx,%rdx), %rcx
	leaq	quote_calc2_table(%rip), %rdx
	movzwl	(%rcx,%rdx), %edx
	movw	%dx, (%rax)
	movq	40+yystack(%rip), %rax
	addq	$4, %rax
	movq	%rax, 40+yystack(%rip)
	movq	40+yystack(%rip), %rax
	movl	quote_calc2_lval(%rip), %edx
	movl	%edx, (%rax)
	movq	$33, -8(%rbp)
	jmp	.L213
.L169:
	movl	quote_calc2_char(%rip), %eax
	testl	%eax, %eax
	jne	.L307
	movq	$133, -8(%rbp)
	jmp	.L213
.L307:
	movq	$33, -8(%rbp)
	jmp	.L213
.L207:
	movl	$0, quote_calc2_errflag(%rip)
	movq	$142, -8(%rbp)
	jmp	.L213
.L189:
	movl	quote_calc2_char(%rip), %eax
	testl	%eax, %eax
	jne	.L309
	movq	$99, -8(%rbp)
	jmp	.L213
.L309:
	movq	$126, -8(%rbp)
	jmp	.L213
.L193:
	movl	$0, %eax
	jmp	.L298
.L125:
	movl	$-1, quote_calc2_char(%rip)
	movq	$33, -8(%rbp)
	jmp	.L213
.L184:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	quote_calc2_error
	movq	$99, -8(%rbp)
	jmp	.L213
.L157:
	movq	8+yystack(%rip), %rax
	movq	%rax, 16+yystack(%rip)
	movq	32+yystack(%rip), %rax
	movq	%rax, 40+yystack(%rip)
	movl	$0, -28(%rbp)
	movq	16+yystack(%rip), %rax
	movw	$0, (%rax)
	movq	$33, -8(%rbp)
	jmp	.L213
.L211:
	movl	-36(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	quote_calc2_dgoto(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	cwtl
	movl	%eax, -28(%rbp)
	movq	$144, -8(%rbp)
	jmp	.L213
.L198:
	movq	16+yystack(%rip), %rax
	movq	%rax, %rdx
	movq	24+yystack(%rip), %rax
	cmpq	%rax, %rdx
	jb	.L311
	movq	$49, -8(%rbp)
	jmp	.L213
.L311:
	movq	$111, -8(%rbp)
	jmp	.L213
.L314:
	nop
.L213:
	jmp	.L313
.L298:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	quote_calc2_parse, .-quote_calc2_parse
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
