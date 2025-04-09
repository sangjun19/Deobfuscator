	.file	"VikaPaz_HW4_C_1_flatten.c"
	.text
	.globl	_TIG_IZ_HHm2_envp
	.bss
	.align 8
	.type	_TIG_IZ_HHm2_envp, @object
	.size	_TIG_IZ_HHm2_envp, 8
_TIG_IZ_HHm2_envp:
	.zero	8
	.globl	_TIG_IZ_HHm2_argv
	.align 8
	.type	_TIG_IZ_HHm2_argv, @object
	.size	_TIG_IZ_HHm2_argv, 8
_TIG_IZ_HHm2_argv:
	.zero	8
	.globl	_TIG_IZ_HHm2_argc
	.align 4
	.type	_TIG_IZ_HHm2_argc, @object
	.size	_TIG_IZ_HHm2_argc, 4
_TIG_IZ_HHm2_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d %d"
.LC1:
	.string	"%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_HHm2_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_HHm2_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_HHm2_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HHm2--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_HHm2_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_HHm2_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_HHm2_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L33:
	cmpq	$16, -16(%rbp)
	ja	.L36
	movq	-16(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L17-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$29, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L21
.L9:
	movl	$28, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L21
.L12:
	movl	$31, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L21
.L16:
	movl	-24(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$5, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$100, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	je	.L22
	movq	$0, -16(%rbp)
	jmp	.L21
.L22:
	movq	$13, -16(%rbp)
	jmp	.L21
.L19:
	movl	$30, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L21
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L13:
	movl	-28(%rbp), %eax
	cmpl	$12, %eax
	seta	%dl
	testb	%dl, %dl
	jne	.L25
	movl	$1, %edx
	movl	%eax, %ecx
	salq	%cl, %rdx
	movq	%rdx, %rax
	movq	%rax, %rdx
	andl	$5546, %edx
	testq	%rdx, %rdx
	setne	%dl
	testb	%dl, %dl
	jne	.L26
	movq	%rax, %rdx
	andl	$2640, %edx
	testq	%rdx, %rdx
	setne	%dl
	testb	%dl, %dl
	jne	.L27
	andl	$4, %eax
	testq	%rax, %rax
	setne	%al
	testb	%al, %al
	je	.L25
	movq	$5, -16(%rbp)
	jmp	.L28
.L27:
	movq	$1, -16(%rbp)
	jmp	.L28
.L26:
	movq	$12, -16(%rbp)
	jmp	.L28
.L25:
	movq	$10, -16(%rbp)
	nop
.L28:
	jmp	.L21
.L15:
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -16(%rbp)
	jmp	.L21
.L11:
	movl	-24(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$7, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$400, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L29
	movq	$14, -16(%rbp)
	jmp	.L21
.L29:
	movq	$15, -16(%rbp)
	jmp	.L21
.L17:
	movl	-24(%rbp), %eax
	andl	$3, %eax
	testl	%eax, %eax
	jne	.L31
	movq	$8, -16(%rbp)
	jmp	.L21
.L31:
	movq	$13, -16(%rbp)
	jmp	.L21
.L14:
	movq	$2, -16(%rbp)
	jmp	.L21
.L20:
	movl	$29, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L21
.L18:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -16(%rbp)
	jmp	.L21
.L36:
	nop
.L21:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
