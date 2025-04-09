	.file	"gabrielcampanile_LP_ex15_flatten.c"
	.text
	.globl	_TIG_IZ_tD9L_envp
	.bss
	.align 8
	.type	_TIG_IZ_tD9L_envp, @object
	.size	_TIG_IZ_tD9L_envp, 8
_TIG_IZ_tD9L_envp:
	.zero	8
	.globl	_TIG_IZ_tD9L_argc
	.align 4
	.type	_TIG_IZ_tD9L_argc, @object
	.size	_TIG_IZ_tD9L_argc, 4
_TIG_IZ_tD9L_argc:
	.zero	4
	.globl	_TIG_IZ_tD9L_argv
	.align 8
	.type	_TIG_IZ_tD9L_argv, @object
	.size	_TIG_IZ_tD9L_argv, 8
_TIG_IZ_tD9L_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Digite um n\303\272mero \303\255mpar: "
.LC1:
	.string	"%i"
	.align 8
.LC2:
	.string	"N\303\272meros \303\255mpares de %i at\303\251 1 em ordem decrescente:\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
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
	movq	$0, _TIG_IZ_tD9L_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_tD9L_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_tD9L_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 100 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-tD9L--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_tD9L_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_tD9L_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_tD9L_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L11:
	cmpq	$2, -16(%rbp)
	je	.L6
	cmpq	$2, -16(%rbp)
	ja	.L14
	cmpq	$0, -16(%rbp)
	je	.L8
	cmpq	$1, -16(%rbp)
	jne	.L14
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	imprima
	movl	$10, %edi
	call	putchar@PLT
	movq	$1, -16(%rbp)
	jmp	.L10
.L6:
	movq	$0, -16(%rbp)
	jmp	.L10
.L14:
	nop
.L10:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC3:
	.string	"%i\t"
	.text
	.globl	imprima
	.type	imprima, @function
imprima:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$3, -8(%rbp)
.L30:
	cmpq	$6, -8(%rbp)
	ja	.L31
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L18(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L18(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L18:
	.long	.L32-.L18
	.long	.L22-.L18
	.long	.L32-.L18
	.long	.L20-.L18
	.long	.L31-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L22:
	movl	-20(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L24
	movq	$5, -8(%rbp)
	jmp	.L26
.L24:
	movq	$6, -8(%rbp)
	jmp	.L26
.L20:
	cmpl	$0, -20(%rbp)
	jg	.L27
	movq	$2, -8(%rbp)
	jmp	.L26
.L27:
	movq	$1, -8(%rbp)
	jmp	.L26
.L17:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edi
	call	imprima
	movq	$0, -8(%rbp)
	jmp	.L26
.L19:
	movl	-20(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edi
	call	imprima
	movq	$0, -8(%rbp)
	jmp	.L26
.L31:
	nop
.L26:
	jmp	.L30
.L32:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	imprima, .-imprima
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
