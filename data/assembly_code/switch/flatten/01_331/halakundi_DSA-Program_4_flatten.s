	.file	"halakundi_DSA-Program_4_flatten.c"
	.text
	.globl	stack
	.bss
	.align 32
	.type	stack, @object
	.size	stack, 40
stack:
	.zero	40
	.globl	i
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.zero	4
	.globl	ele
	.align 4
	.type	ele, @object
	.size	ele, 4
ele:
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
	.globl	_TIG_IZ_YgeX_argv
	.align 8
	.type	_TIG_IZ_YgeX_argv, @object
	.size	_TIG_IZ_YgeX_argv, 8
_TIG_IZ_YgeX_argv:
	.zero	8
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.globl	_TIG_IZ_YgeX_envp
	.align 8
	.type	_TIG_IZ_YgeX_envp, @object
	.size	_TIG_IZ_YgeX_envp, 8
_TIG_IZ_YgeX_envp:
	.zero	8
	.globl	sym
	.type	sym, @object
	.size	sym, 1
sym:
	.zero	1
	.globl	_TIG_IZ_YgeX_argc
	.align 4
	.type	_TIG_IZ_YgeX_argc, @object
	.size	_TIG_IZ_YgeX_argc, 4
_TIG_IZ_YgeX_argc:
	.zero	4
	.text
	.globl	eval
	.type	eval, @function
eval:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, %eax
	movl	%edx, -44(%rbp)
	movb	%al, -40(%rbp)
	movq	$3, -16(%rbp)
.L21:
	cmpq	$15, -16(%rbp)
	ja	.L22
	movq	-16(%rbp), %rax
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
	.long	.L11-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L8-.L4
	.long	.L22-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L3-.L4
	.text
.L9:
	movq	$13, -16(%rbp)
	jmp	.L12
.L3:
	movl	-36(%rbp), %eax
	imull	-44(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -16(%rbp)
	jmp	.L12
.L6:
	movl	-36(%rbp), %eax
	cltd
	idivl	-44(%rbp)
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -16(%rbp)
	jmp	.L12
.L10:
	movsbl	-40(%rbp), %eax
	cmpl	$94, %eax
	je	.L13
	cmpl	$94, %eax
	jg	.L14
	cmpl	$47, %eax
	je	.L15
	cmpl	$47, %eax
	jg	.L14
	cmpl	$45, %eax
	je	.L16
	cmpl	$45, %eax
	jg	.L14
	cmpl	$42, %eax
	je	.L17
	cmpl	$43, %eax
	je	.L18
	jmp	.L14
.L13:
	movq	$11, -16(%rbp)
	jmp	.L19
.L15:
	movq	$12, -16(%rbp)
	jmp	.L19
.L17:
	movq	$15, -16(%rbp)
	jmp	.L19
.L16:
	movq	$0, -16(%rbp)
	jmp	.L19
.L18:
	movq	$9, -16(%rbp)
	jmp	.L19
.L14:
	movq	$4, -16(%rbp)
	nop
.L19:
	jmp	.L12
.L7:
	pxor	%xmm0, %xmm0
	cvtsi2sdl	-44(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtsi2sdl	-36(%rbp), %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movsd	-8(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -16(%rbp)
	jmp	.L12
.L8:
	movl	-36(%rbp), %edx
	movl	-44(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -16(%rbp)
	jmp	.L12
.L11:
	movl	-36(%rbp), %eax
	subl	-44(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -16(%rbp)
	jmp	.L12
.L22:
	nop
.L12:
	jmp	.L21
.L23:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	eval, .-eval
	.section	.rodata
.LC0:
	.string	"Result = %d\n"
	.align 8
.LC1:
	.string	"Enter the valid postfix expression:\t"
.LC2:
	.string	"%s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movb	$0, sym(%rip)
	nop
.L25:
	movl	$0, op2(%rip)
	nop
.L26:
	movl	$0, op1(%rip)
	nop
.L27:
	movl	$0, ele(%rip)
	nop
.L28:
	movl	$0, i(%rip)
	nop
.L29:
	movl	$-1, top(%rip)
	nop
.L30:
	movl	$0, -56(%rbp)
	jmp	.L31
.L32:
	movl	-56(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -56(%rbp)
.L31:
	cmpl	$9, -56(%rbp)
	jle	.L32
	nop
.L33:
	movq	$0, _TIG_IZ_YgeX_envp(%rip)
	nop
.L34:
	movq	$0, _TIG_IZ_YgeX_argv(%rip)
	nop
.L35:
	movl	$0, _TIG_IZ_YgeX_argc(%rip)
	nop
	nop
.L36:
.L37:
#APP
# 177 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-YgeX--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_YgeX_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_YgeX_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_YgeX_envp(%rip)
	nop
	movq	$9, -40(%rbp)
.L56:
	cmpq	$14, -40(%rbp)
	ja	.L59
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L49-.L40
	.long	.L48-.L40
	.long	.L59-.L40
	.long	.L47-.L40
	.long	.L59-.L40
	.long	.L46-.L40
	.long	.L59-.L40
	.long	.L45-.L40
	.long	.L59-.L40
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L42-.L40
	.long	.L41-.L40
	.long	.L59-.L40
	.long	.L39-.L40
	.text
.L39:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	sym(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L50
	movq	$10, -40(%rbp)
	jmp	.L52
.L50:
	movq	$12, -40(%rbp)
	jmp	.L52
.L41:
	call	pop
	movl	%eax, op2(%rip)
	call	pop
	movl	%eax, op1(%rip)
	movl	op2(%rip), %edx
	movzbl	sym(%rip), %eax
	movsbl	%al, %ecx
	movl	op1(%rip), %eax
	movl	%ecx, %esi
	movl	%eax, %edi
	call	eval
	movq	$11, -40(%rbp)
	jmp	.L52
.L48:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L57
	jmp	.L58
.L47:
	movl	i(%rip), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	testb	%al, %al
	je	.L54
	movq	$5, -40(%rbp)
	jmp	.L52
.L54:
	movq	$0, -40(%rbp)
	jmp	.L52
.L42:
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$3, -40(%rbp)
	jmp	.L52
.L44:
	movq	$7, -40(%rbp)
	jmp	.L52
.L46:
	movl	i(%rip), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movb	%al, sym(%rip)
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$14, -40(%rbp)
	jmp	.L52
.L43:
	movzbl	sym(%rip), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	movl	%eax, %edi
	call	push
	movq	$11, -40(%rbp)
	jmp	.L52
.L49:
	call	pop
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -40(%rbp)
	jmp	.L52
.L45:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, i(%rip)
	movq	$3, -40(%rbp)
	jmp	.L52
.L59:
	nop
.L52:
	jmp	.L56
.L58:
	call	__stack_chk_fail@PLT
.L57:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	pop
	.type	pop, @function
pop:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L66:
	cmpq	$2, -8(%rbp)
	je	.L61
	cmpq	$2, -8(%rbp)
	ja	.L68
	cmpq	$0, -8(%rbp)
	je	.L63
	cmpq	$1, -8(%rbp)
	jne	.L68
	movq	$2, -8(%rbp)
	jmp	.L64
.L63:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L67
.L61:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$0, -8(%rbp)
	jmp	.L64
.L68:
	nop
.L64:
	jmp	.L66
.L67:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	pop, .-pop
	.globl	push
	.type	push, @function
push:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L75:
	cmpq	$2, -8(%rbp)
	je	.L70
	cmpq	$2, -8(%rbp)
	ja	.L77
	cmpq	$0, -8(%rbp)
	je	.L72
	cmpq	$1, -8(%rbp)
	jne	.L77
	jmp	.L76
.L72:
	movq	$2, -8(%rbp)
	jmp	.L74
.L70:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	stack(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$1, -8(%rbp)
	jmp	.L74
.L77:
	nop
.L74:
	jmp	.L75
.L76:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	push, .-push
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
