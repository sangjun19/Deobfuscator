	.file	"YaraAshraf22_Learn-in-Depth-Diploma_main_flatten.c"
	.text
	.globl	_TIG_IZ_kW10_argv
	.bss
	.align 8
	.type	_TIG_IZ_kW10_argv, @object
	.size	_TIG_IZ_kW10_argv, 8
_TIG_IZ_kW10_argv:
	.zero	8
	.globl	_TIG_IZ_kW10_envp
	.align 8
	.type	_TIG_IZ_kW10_envp, @object
	.size	_TIG_IZ_kW10_envp, 8
_TIG_IZ_kW10_envp:
	.zero	8
	.globl	_TIG_IZ_kW10_argc
	.align 4
	.type	_TIG_IZ_kW10_argc, @object
	.size	_TIG_IZ_kW10_argc, 4
_TIG_IZ_kW10_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter no of elements : "
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"\nEnter the element to be searched: "
.LC3:
	.string	"\nNumber not found"
	.align 8
.LC4:
	.string	"\nNumber found at location = %d"
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
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_kW10_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_kW10_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_kW10_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-kW10--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_kW10_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_kW10_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_kW10_envp(%rip)
	nop
	movq	$9, -136(%rbp)
.L33:
	cmpq	$17, -136(%rbp)
	ja	.L36
	movq	-136(%rbp), %rax
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
	.long	.L36-.L8
	.long	.L20-.L8
	.long	.L36-.L8
	.long	.L19-.L8
	.long	.L36-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L36-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L36-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	-140(%rbp), %eax
	cltq
	movl	-128(%rbp,%rax,4), %edx
	movl	-144(%rbp), %eax
	cmpl	%eax, %edx
	je	.L21
	movq	$17, -136(%rbp)
	jmp	.L23
.L21:
	movq	$5, -136(%rbp)
	jmp	.L23
.L9:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	-148(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -140(%rbp)
	movq	$11, -136(%rbp)
	jmp	.L23
.L12:
	movl	-148(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L24
	movq	$7, -136(%rbp)
	jmp	.L23
.L24:
	movq	$3, -136(%rbp)
	jmp	.L23
.L20:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	-144(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -140(%rbp)
	movq	$12, -136(%rbp)
	jmp	.L23
.L19:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L13:
	movl	-148(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L27
	movq	$6, -136(%rbp)
	jmp	.L23
.L27:
	movq	$1, -136(%rbp)
	jmp	.L23
.L15:
	movq	$15, -136(%rbp)
	jmp	.L23
.L11:
	movl	-148(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -140(%rbp)
	jne	.L29
	movq	$14, -136(%rbp)
	jmp	.L23
.L29:
	movq	$5, -136(%rbp)
	jmp	.L23
.L7:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -136(%rbp)
	jmp	.L23
.L17:
	leaq	-128(%rbp), %rdx
	movl	-140(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -140(%rbp)
	movq	$11, -136(%rbp)
	jmp	.L23
.L18:
	addl	$1, -140(%rbp)
	movq	$12, -136(%rbp)
	jmp	.L23
.L14:
	movl	-140(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -136(%rbp)
	jmp	.L23
.L16:
	movl	-140(%rbp), %eax
	cltq
	movl	-128(%rbp,%rax,4), %edx
	movl	-144(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L31
	movq	$10, -136(%rbp)
	jmp	.L23
.L31:
	movq	$13, -136(%rbp)
	jmp	.L23
.L36:
	nop
.L23:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
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
