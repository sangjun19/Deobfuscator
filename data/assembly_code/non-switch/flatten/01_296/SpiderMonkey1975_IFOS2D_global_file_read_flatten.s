	.file	"SpiderMonkey1975_IFOS2D_global_file_read_flatten.c"
	.text
	.globl	_TIG_IZ_qqt1_envp
	.bss
	.align 8
	.type	_TIG_IZ_qqt1_envp, @object
	.size	_TIG_IZ_qqt1_envp, 8
_TIG_IZ_qqt1_envp:
	.zero	8
	.globl	_TIG_IZ_qqt1_argv
	.align 8
	.type	_TIG_IZ_qqt1_argv, @object
	.size	_TIG_IZ_qqt1_argv, 8
_TIG_IZ_qqt1_argv:
	.zero	8
	.globl	_TIG_IZ_qqt1_argc
	.align 4
	.type	_TIG_IZ_qqt1_argc, @object
	.size	_TIG_IZ_qqt1_argc, 4
_TIG_IZ_qqt1_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"rb"
.LC1:
	.string	"%f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, _TIG_IZ_qqt1_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_qqt1_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_qqt1_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 124 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qqt1--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_qqt1_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_qqt1_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_qqt1_envp(%rip)
	nop
	movq	$5, -24(%rbp)
.L17:
	cmpq	$7, -24(%rbp)
	ja	.L19
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L19-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L19-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L19-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$12288, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-64(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rcx
	movl	$4, %edx
	movl	$3072, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	$0, -36(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L13
.L12:
	cmpl	$3071, -36(%rbp)
	jg	.L14
	movq	$2, -24(%rbp)
	jmp	.L13
.L14:
	movq	$7, -24(%rbp)
	jmp	.L13
.L9:
	movq	$4, -24(%rbp)
	jmp	.L13
.L7:
	movl	$0, %eax
	jmp	.L18
.L11:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -36(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L13
.L19:
	nop
.L13:
	jmp	.L17
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
