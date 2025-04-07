	.file	"LyndaLilly_alx-low_level_programming_9-fizz_buzz_flatten.c"
	.text
	.globl	_TIG_IZ_vxCJ_envp
	.bss
	.align 8
	.type	_TIG_IZ_vxCJ_envp, @object
	.size	_TIG_IZ_vxCJ_envp, 8
_TIG_IZ_vxCJ_envp:
	.zero	8
	.globl	_TIG_IZ_vxCJ_argc
	.align 4
	.type	_TIG_IZ_vxCJ_argc, @object
	.size	_TIG_IZ_vxCJ_argc, 4
_TIG_IZ_vxCJ_argc:
	.zero	4
	.globl	_TIG_IZ_vxCJ_argv
	.align 8
	.type	_TIG_IZ_vxCJ_argv, @object
	.size	_TIG_IZ_vxCJ_argv, 8
_TIG_IZ_vxCJ_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"Fizz"
.LC2:
	.string	"Buzz"
.LC3:
	.string	"FizzBuzz"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_vxCJ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_vxCJ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_vxCJ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-vxCJ--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_vxCJ_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_vxCJ_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_vxCJ_envp(%rip)
	nop
	movq	$18, -8(%rbp)
.L37:
	cmpq	$18, -8(%rbp)
	ja	.L39
	movq	-8(%rbp), %rax
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
	.long	.L39-.L8
	.long	.L22-.L8
	.long	.L39-.L8
	.long	.L21-.L8
	.long	.L39-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L39-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L23
.L11:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -8(%rbp)
	jmp	.L23
.L10:
	cmpl	$100, -12(%rbp)
	jne	.L24
	movq	$11, -8(%rbp)
	jmp	.L23
.L24:
	movq	$3, -8(%rbp)
	jmp	.L23
.L13:
	movl	$0, %eax
	jmp	.L38
.L17:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -8(%rbp)
	jmp	.L23
.L22:
	movl	-12(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1431655766, %rax, %rax
	shrq	$32, %rax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	addl	%ecx, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L27
	movq	$7, -8(%rbp)
	jmp	.L23
.L27:
	movq	$6, -8(%rbp)
	jmp	.L23
.L21:
	movl	$32, %edi
	call	putchar@PLT
	movq	$11, -8(%rbp)
	jmp	.L23
.L14:
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L23
.L16:
	movl	$10, %edi
	call	putchar@PLT
	movq	$12, -8(%rbp)
	jmp	.L23
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -8(%rbp)
	jmp	.L23
.L9:
	movl	-12(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	%eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$2, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L29
	movq	$13, -8(%rbp)
	jmp	.L23
.L29:
	movq	$14, -8(%rbp)
	jmp	.L23
.L19:
	movl	-12(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1431655766, %rax, %rax
	shrq	$32, %rax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	addl	%ecx, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L31
	movq	$8, -8(%rbp)
	jmp	.L23
.L31:
	movq	$17, -8(%rbp)
	jmp	.L23
.L20:
	cmpl	$100, -12(%rbp)
	jg	.L33
	movq	$1, -8(%rbp)
	jmp	.L23
.L33:
	movq	$9, -8(%rbp)
	jmp	.L23
.L15:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -8(%rbp)
	jmp	.L23
.L18:
	movl	-12(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	%eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$2, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L35
	movq	$10, -8(%rbp)
	jmp	.L23
.L35:
	movq	$6, -8(%rbp)
	jmp	.L23
.L39:
	nop
.L23:
	jmp	.L37
.L38:
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
