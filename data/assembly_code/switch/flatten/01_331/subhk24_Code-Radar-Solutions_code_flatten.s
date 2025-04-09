	.file	"subhk24_Code-Radar-Solutions_code_flatten.c"
	.text
	.globl	_TIG_IZ_fv26_argc
	.bss
	.align 4
	.type	_TIG_IZ_fv26_argc, @object
	.size	_TIG_IZ_fv26_argc, 4
_TIG_IZ_fv26_argc:
	.zero	4
	.globl	_TIG_IZ_fv26_argv
	.align 8
	.type	_TIG_IZ_fv26_argv, @object
	.size	_TIG_IZ_fv26_argv, 8
_TIG_IZ_fv26_argv:
	.zero	8
	.globl	_TIG_IZ_fv26_envp
	.align 8
	.type	_TIG_IZ_fv26_envp, @object
	.size	_TIG_IZ_fv26_envp, 8
_TIG_IZ_fv26_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Consonant"
.LC1:
	.string	"Special Character"
.LC2:
	.string	"Digit"
.LC3:
	.string	"Vowel"
.LC4:
	.string	"%c"
.LC5:
	.string	"Invalid input"
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
	movq	$0, _TIG_IZ_fv26_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_fv26_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_fv26_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fv26--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_fv26_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_fv26_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_fv26_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L26:
	cmpq	$9, -16(%rbp)
	ja	.L30
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
	.long	.L30-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L30-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L16
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L16
.L15:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L16
.L7:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L16
.L11:
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -16(%rbp)
	jmp	.L16
.L12:
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$122, %eax
	jg	.L17
	cmpl	$63, %eax
	jge	.L18
	cmpl	$38, %eax
	jg	.L19
	cmpl	$36, %eax
	jge	.L20
	jmp	.L17
.L18:
	subl	$63, %eax
	movl	$1, %edx
	movl	%eax, %ecx
	salq	%cl, %rdx
	movq	%rdx, %rax
	movabsq	$1134620923836497920, %rdx
	andq	%rax, %rdx
	testq	%rdx, %rdx
	setne	%dl
	testb	%dl, %dl
	jne	.L21
	movabsq	$18300563590545408, %rdx
	andq	%rax, %rdx
	testq	%rdx, %rdx
	setne	%dl
	testb	%dl, %dl
	jne	.L22
	andl	$3, %eax
	testq	%rax, %rax
	setne	%al
	testb	%al, %al
	jne	.L20
	jmp	.L17
.L19:
	subl	$48, %eax
	cmpl	$9, %eax
	ja	.L17
	jmp	.L28
.L20:
	movq	$8, -16(%rbp)
	jmp	.L24
.L28:
	movq	$1, -16(%rbp)
	jmp	.L24
.L21:
	movq	$4, -16(%rbp)
	jmp	.L24
.L22:
	movq	$9, -16(%rbp)
	jmp	.L24
.L17:
	movq	$2, -16(%rbp)
	nop
.L24:
	jmp	.L16
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	jmp	.L29
.L14:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L16
.L30:
	nop
.L16:
	jmp	.L26
.L29:
	call	__stack_chk_fail@PLT
.L27:
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
