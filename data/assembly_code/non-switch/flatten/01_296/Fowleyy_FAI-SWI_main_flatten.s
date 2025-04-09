	.file	"Fowleyy_FAI-SWI_main_flatten.c"
	.text
	.globl	_TIG_IZ_sTPF_argv
	.bss
	.align 8
	.type	_TIG_IZ_sTPF_argv, @object
	.size	_TIG_IZ_sTPF_argv, 8
_TIG_IZ_sTPF_argv:
	.zero	8
	.globl	_TIG_IZ_sTPF_envp
	.align 8
	.type	_TIG_IZ_sTPF_envp, @object
	.size	_TIG_IZ_sTPF_envp, 8
_TIG_IZ_sTPF_envp:
	.zero	8
	.globl	_TIG_IZ_sTPF_argc
	.align 4
	.type	_TIG_IZ_sTPF_argc, @object
	.size	_TIG_IZ_sTPF_argc, 4
_TIG_IZ_sTPF_argc:
	.zero	4
	.text
	.globl	df
	.type	df, @function
df:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movq	$0, -8(%rbp)
.L4:
	cmpq	$0, -8(%rbp)
	jne	.L7
	movss	-20(%rbp), %xmm0
	addss	%xmm0, %xmm0
	jmp	.L6
.L7:
	nop
	jmp	.L4
.L6:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	df, .-df
	.globl	newton_runn
	.type	newton_runn, @function
newton_runn:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movss	%xmm0, -24(%rbp)
	movq	$4, -8(%rbp)
.L18:
	cmpq	$4, -8(%rbp)
	je	.L9
	cmpq	$4, -8(%rbp)
	ja	.L19
	cmpq	$3, -8(%rbp)
	je	.L11
	cmpq	$3, -8(%rbp)
	ja	.L19
	cmpq	$0, -8(%rbp)
	je	.L12
	cmpq	$1, -8(%rbp)
	je	.L13
	jmp	.L19
.L9:
	cmpl	$0, -20(%rbp)
	jne	.L14
	movq	$3, -8(%rbp)
	jmp	.L16
.L14:
	movq	$0, -8(%rbp)
	jmp	.L16
.L13:
	movss	-16(%rbp), %xmm0
	jmp	.L17
.L11:
	movss	-24(%rbp), %xmm0
	jmp	.L17
.L12:
	movl	-20(%rbp), %eax
	leal	-1(%rax), %edx
	movl	-24(%rbp), %eax
	movd	%eax, %xmm0
	movl	%edx, %edi
	call	newton_runn
	movd	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movd	%eax, %xmm0
	call	newton
	movd	%xmm0, %eax
	movl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L16
.L19:
	nop
.L16:
	jmp	.L18
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	newton_runn, .-newton_runn
	.section	.rodata
.LC0:
	.string	"konec"
.LC2:
	.string	"times - %d\n"
.LC3:
	.string	"val - %lf\n"
	.text
	.globl	newton_run
	.type	newton_run, @function
newton_run:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movss	%xmm0, -40(%rbp)
	movss	%xmm1, -44(%rbp)
	movq	$9, -8(%rbp)
.L38:
	cmpq	$9, -8(%rbp)
	ja	.L40
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L30-.L23
	.long	.L29-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L40-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L40-.L23
	.long	.L22-.L23
	.text
.L26:
	movss	-24(%rbp), %xmm0
	jmp	.L31
.L29:
	movl	-44(%rbp), %eax
	movd	%eax, %xmm0
	call	newton
	movd	%xmm0, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movd	%eax, %xmm0
	call	newton
	movd	%xmm0, %eax
	movl	%eax, -16(%rbp)
	movl	-40(%rbp), %eax
	movd	%eax, %xmm0
	call	newton
	movd	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movl	-36(%rbp), %eax
	leal	-1(%rax), %edx
	movss	-16(%rbp), %xmm0
	movl	-12(%rbp), %eax
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	movl	%edx, %edi
	call	newton_run
	movd	%xmm0, %eax
	movl	%eax, -24(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L32
.L27:
	movss	-40(%rbp), %xmm0
	ucomiss	-44(%rbp), %xmm0
	jp	.L33
	movss	-40(%rbp), %xmm0
	ucomiss	-44(%rbp), %xmm0
	jne	.L33
	movq	$0, -8(%rbp)
	jmp	.L32
.L33:
	movq	$1, -8(%rbp)
	jmp	.L32
.L22:
	cmpl	$0, -36(%rbp)
	jne	.L36
	movq	$7, -8(%rbp)
	jmp	.L32
.L36:
	movq	$2, -8(%rbp)
	jmp	.L32
.L25:
	movss	-40(%rbp), %xmm0
	jmp	.L31
.L30:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -8(%rbp)
	jmp	.L32
.L24:
	pxor	%xmm0, %xmm0
	jmp	.L31
.L28:
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	pxor	%xmm2, %xmm2
	cvtss2sd	-40(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L32
.L40:
	nop
.L32:
	jmp	.L38
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	newton_run, .-newton_run
	.globl	newton
	.type	newton, @function
newton:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movq	$2, -8(%rbp)
.L47:
	cmpq	$2, -8(%rbp)
	je	.L42
	cmpq	$2, -8(%rbp)
	ja	.L49
	cmpq	$0, -8(%rbp)
	je	.L44
	cmpq	$1, -8(%rbp)
	jne	.L49
	movss	-16(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	divss	-12(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	subss	%xmm1, %xmm0
	jmp	.L48
.L44:
	movl	-20(%rbp), %eax
	movd	%eax, %xmm0
	call	f
	movd	%xmm0, %eax
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
	movd	%eax, %xmm0
	call	df
	movd	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L46
.L42:
	movq	$0, -8(%rbp)
	jmp	.L46
.L49:
	nop
.L46:
	jmp	.L47
.L48:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	newton, .-newton
	.globl	f
	.type	f, @function
f:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movq	$0, -8(%rbp)
.L53:
	cmpq	$0, -8(%rbp)
	jne	.L56
	movss	-20(%rbp), %xmm0
	mulss	%xmm0, %xmm0
	movss	.LC4(%rip), %xmm1
	subss	%xmm1, %xmm0
	jmp	.L55
.L56:
	nop
	jmp	.L53
.L55:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	f, .-f
	.section	.rodata
.LC6:
	.string	"%lf"
	.text
	.globl	main
	.type	main, @function
main:
.LFB10:
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
	movq	$0, _TIG_IZ_sTPF_envp(%rip)
	nop
.L58:
	movq	$0, _TIG_IZ_sTPF_argv(%rip)
	nop
.L59:
	movl	$0, _TIG_IZ_sTPF_argc(%rip)
	nop
	nop
.L60:
.L61:
#APP
# 89 "Fowleyy_FAI-SWI_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-sTPF--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_sTPF_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_sTPF_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_sTPF_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L67:
	cmpq	$2, -8(%rbp)
	je	.L62
	cmpq	$2, -8(%rbp)
	ja	.L69
	cmpq	$0, -8(%rbp)
	je	.L64
	cmpq	$1, -8(%rbp)
	jne	.L69
	movl	$0, %eax
	jmp	.L68
.L64:
	movl	.LC5(%rip), %eax
	movd	%eax, %xmm0
	call	newton
	movd	%xmm0, %eax
	movl	%eax, -16(%rbp)
	movss	-16(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	movl	.LC5(%rip), %eax
	movd	%eax, %xmm0
	movl	$1000, %edi
	call	newton_run
	movd	%xmm0, %eax
	movl	%eax, -12(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-12(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L66
.L62:
	movq	$0, -8(%rbp)
	jmp	.L66
.L69:
	nop
.L66:
	jmp	.L67
.L68:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC4:
	.long	1084227584
	.align 4
.LC5:
	.long	1148846080
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
