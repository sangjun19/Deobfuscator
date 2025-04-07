	.file	"Dhruvanarayana_DSA-LAB_5_flatten.c"
	.text
	.globl	_TIG_IZ_dHvb_envp
	.bss
	.align 8
	.type	_TIG_IZ_dHvb_envp, @object
	.size	_TIG_IZ_dHvb_envp, 8
_TIG_IZ_dHvb_envp:
	.zero	8
	.globl	i
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.zero	4
	.globl	s
	.align 32
	.type	s, @object
	.size	s, 80
s:
	.zero	80
	.globl	_TIG_IZ_dHvb_argc
	.align 4
	.type	_TIG_IZ_dHvb_argc, @object
	.size	_TIG_IZ_dHvb_argc, 4
_TIG_IZ_dHvb_argc:
	.zero	4
	.globl	op1
	.align 4
	.type	op1, @object
	.size	op1, 4
op1:
	.zero	4
	.globl	op2
	.align 4
	.type	op2, @object
	.size	op2, 4
op2:
	.zero	4
	.globl	postfix
	.align 32
	.type	postfix, @object
	.size	postfix, 90
postfix:
	.zero	90
	.globl	res
	.align 4
	.type	res, @object
	.size	res, 4
res:
	.zero	4
	.globl	symb
	.type	symb, @object
	.size	symb, 1
symb:
	.zero	1
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.globl	_TIG_IZ_dHvb_argv
	.align 8
	.type	_TIG_IZ_dHvb_argv, @object
	.size	_TIG_IZ_dHvb_argv, 8
_TIG_IZ_dHvb_argv:
	.zero	8
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movl	-16(%rbp), %eax
	jmp	.L8
.L4:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L6
.L2:
	movq	$0, -8(%rbp)
	jmp	.L6
.L9:
	nop
.L6:
	jmp	.L7
.L8:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	pop, .-pop
	.globl	push
	.type	push, @function
push:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L16:
	cmpq	$2, -8(%rbp)
	je	.L11
	cmpq	$2, -8(%rbp)
	ja	.L17
	cmpq	$0, -8(%rbp)
	je	.L18
	cmpq	$1, -8(%rbp)
	jne	.L17
	movq	$2, -8(%rbp)
	jmp	.L14
.L11:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	s(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$0, -8(%rbp)
	jmp	.L14
.L17:
	nop
.L14:
	jmp	.L16
.L18:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	push, .-push
	.section	.rodata
.LC0:
	.string	"\n Result=%d"
	.align 8
.LC1:
	.string	"Enter a valid postfix expression:"
.LC2:
	.string	"%s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movb	$0, symb(%rip)
	nop
.L20:
	movl	$0, -32(%rbp)
	jmp	.L21
.L22:
	movl	-32(%rbp), %eax
	cltq
	leaq	postfix(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -32(%rbp)
.L21:
	cmpl	$89, -32(%rbp)
	jle	.L22
	nop
.L23:
	movl	$0, -28(%rbp)
	jmp	.L24
.L25:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -28(%rbp)
.L24:
	cmpl	$19, -28(%rbp)
	jle	.L25
	nop
.L26:
	movl	$0, res(%rip)
	nop
.L27:
	movl	$0, op2(%rip)
	nop
.L28:
	movl	$0, op1(%rip)
	nop
.L29:
	movl	$-1, top(%rip)
	nop
.L30:
	movl	$0, i(%rip)
	nop
.L31:
	movq	$0, _TIG_IZ_dHvb_envp(%rip)
	nop
.L32:
	movq	$0, _TIG_IZ_dHvb_argv(%rip)
	nop
.L33:
	movl	$0, _TIG_IZ_dHvb_argc(%rip)
	nop
	nop
.L34:
.L35:
#APP
# 191 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-dHvb--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_dHvb_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_dHvb_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_dHvb_envp(%rip)
	nop
	movq	$15, -16(%rbp)
.L71:
	cmpq	$29, -16(%rbp)
	ja	.L72
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L38(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L38(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L38:
	.long	.L72-.L38
	.long	.L55-.L38
	.long	.L72-.L38
	.long	.L54-.L38
	.long	.L53-.L38
	.long	.L52-.L38
	.long	.L72-.L38
	.long	.L51-.L38
	.long	.L50-.L38
	.long	.L72-.L38
	.long	.L49-.L38
	.long	.L48-.L38
	.long	.L47-.L38
	.long	.L72-.L38
	.long	.L72-.L38
	.long	.L46-.L38
	.long	.L72-.L38
	.long	.L45-.L38
	.long	.L72-.L38
	.long	.L72-.L38
	.long	.L44-.L38
	.long	.L43-.L38
	.long	.L73-.L38
	.long	.L72-.L38
	.long	.L72-.L38
	.long	.L41-.L38
	.long	.L40-.L38
	.long	.L72-.L38
	.long	.L39-.L38
	.long	.L37-.L38
	.text
.L41:
	movl	i(%rip), %eax
	cltq
	leaq	postfix(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movb	%al, symb(%rip)
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$28, -16(%rbp)
	jmp	.L56
.L53:
	call	pop
	movl	%eax, op2(%rip)
	call	pop
	movl	%eax, op1(%rip)
	movq	$29, -16(%rbp)
	jmp	.L56
.L46:
	movq	$10, -16(%rbp)
	jmp	.L56
.L47:
	movl	op2(%rip), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	movl	op1(%rip), %eax
	pxor	%xmm2, %xmm2
	cvtsi2sdl	%eax, %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movsd	-8(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L50:
	movl	op1(%rip), %eax
	movl	op2(%rip), %edx
	subl	%edx, %eax
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L55:
	movl	i(%rip), %eax
	cltq
	leaq	postfix(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	testb	%al, %al
	je	.L57
	movq	$25, -16(%rbp)
	jmp	.L56
.L57:
	movq	$21, -16(%rbp)
	jmp	.L56
.L54:
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$1, -16(%rbp)
	jmp	.L56
.L43:
	call	pop
	movl	%eax, res(%rip)
	movl	res(%rip), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -16(%rbp)
	jmp	.L56
.L40:
	movl	op1(%rip), %eax
	movl	op2(%rip), %ecx
	cltd
	idivl	%ecx
	movl	%edx, %eax
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L48:
	movzbl	symb(%rip), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L45:
	movl	op1(%rip), %edx
	movl	op2(%rip), %eax
	addl	%edx, %eax
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L39:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	symb(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L60
	movq	$11, -16(%rbp)
	jmp	.L56
.L60:
	movq	$4, -16(%rbp)
	jmp	.L56
.L52:
	movl	op1(%rip), %eax
	movl	op2(%rip), %ecx
	cltd
	idivl	%ecx
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L49:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	postfix(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, i(%rip)
	movq	$1, -16(%rbp)
	jmp	.L56
.L51:
	movl	$0, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L37:
	movzbl	symb(%rip), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	jg	.L62
	cmpl	$37, %eax
	jl	.L63
	subl	$37, %eax
	cmpl	$10, %eax
	ja	.L63
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L65(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L65(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L65:
	.long	.L69-.L65
	.long	.L63-.L65
	.long	.L63-.L65
	.long	.L63-.L65
	.long	.L63-.L65
	.long	.L68-.L65
	.long	.L67-.L65
	.long	.L63-.L65
	.long	.L66-.L65
	.long	.L63-.L65
	.long	.L64-.L65
	.text
.L62:
	cmpl	$94, %eax
	jne	.L63
	movq	$12, -16(%rbp)
	jmp	.L70
.L69:
	movq	$26, -16(%rbp)
	jmp	.L70
.L64:
	movq	$5, -16(%rbp)
	jmp	.L70
.L68:
	movq	$20, -16(%rbp)
	jmp	.L70
.L66:
	movq	$8, -16(%rbp)
	jmp	.L70
.L67:
	movq	$17, -16(%rbp)
	jmp	.L70
.L63:
	movq	$7, -16(%rbp)
	nop
.L70:
	jmp	.L56
.L44:
	movl	op1(%rip), %edx
	movl	op2(%rip), %eax
	imull	%edx, %eax
	movl	%eax, %edi
	call	push
	movq	$3, -16(%rbp)
	jmp	.L56
.L72:
	nop
.L56:
	jmp	.L71
.L73:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
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
