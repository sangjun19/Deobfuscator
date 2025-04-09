	.file	"biswasanimesh_C_Code_Solving_Switch87_flatten.c"
	.text
	.globl	_TIG_IZ_lTaZ_envp
	.bss
	.align 8
	.type	_TIG_IZ_lTaZ_envp, @object
	.size	_TIG_IZ_lTaZ_envp, 8
_TIG_IZ_lTaZ_envp:
	.zero	8
	.globl	_TIG_IZ_lTaZ_argc
	.align 4
	.type	_TIG_IZ_lTaZ_argc, @object
	.size	_TIG_IZ_lTaZ_argc, 4
_TIG_IZ_lTaZ_argc:
	.zero	4
	.globl	_TIG_IZ_lTaZ_argv
	.align 8
	.type	_TIG_IZ_lTaZ_argv, @object
	.size	_TIG_IZ_lTaZ_argv, 8
_TIG_IZ_lTaZ_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter any numbers : "
.LC1:
	.string	"%c"
.LC2:
	.string	"Vowel"
.LC3:
	.string	"Consonant"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_lTaZ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_lTaZ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_lTaZ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-lTaZ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_lTaZ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_lTaZ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_lTaZ_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L18:
	cmpq	$6, -16(%rbp)
	ja	.L21
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
	.long	.L21-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L14
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L7:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L9:
	movq	$4, -16(%rbp)
	jmp	.L14
.L12:
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	cmpl	$52, %eax
	seta	%dl
	testb	%dl, %dl
	jne	.L16
	movabsq	$4575140898685201, %rdx
	movl	%eax, %ecx
	shrq	%cl, %rdx
	movq	%rdx, %rax
	andl	$1, %eax
	testq	%rax, %rax
	setne	%al
	testb	%al, %al
	je	.L16
	movq	$1, -16(%rbp)
	jmp	.L17
.L16:
	movq	$6, -16(%rbp)
	nop
.L17:
	jmp	.L14
.L21:
	nop
.L14:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
